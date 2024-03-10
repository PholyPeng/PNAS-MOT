from functools import partial
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.gcn import affinity_module
from modules.new_end import *  # noqa
from modules.score_net import *  # noqa
from modules.genotypes import PRIMITIVES, Genotype
from modules.operations import *
from solvers import ortools_solve
from utils.data_util import get_start_gt_anno
from modules.transfuser import ImageCNN, LidarEncoder
import pickle


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._latency_list = []
        with open('./latency/lut.pkl', 'rb') as f:
            # dict_keys(['none', 'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3', 'dil_conv_5x5', 'conv_7x1_1x7'])
            latency_dict = pickle.load(f)
            for primitive in PRIMITIVES:
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive or primitive == 'none':
                    op_params = 'in_channels_{}_out_channels_{}_stride_{}'.format(str(C), str(C), str(stride))
                else:
                    op_params = 'in_channels_{}_out_channels_{}_stride_{}_affine_{}'.format(str(C), str(C), str(stride), 'False')
                latency = latency_dict[primitive][op_params]
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
                self._latency_list.append(latency)
        assert len(self._latency_list) == len(self._ops)
        self._latency_list = np.array(self._latency_list, dtype = np.float)
        

    def forward(self, x, weights):
        self.lat = sum(w * lat for w, lat in zip(weights, self._latency_list))
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev,
                                          C,
                                          1,
                                          1,
                                          0,
                                          affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        self.cell_lat = 0.0
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            self.cell_lat += sum(self._ops[offset + j].lat for j in range(len(states)))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

class TrackingNetwork(nn.Module):
    def __init__(self,
                 C_image = 64,
                 C_lidar = 64,
                 out_channels = 512,
                 layers_image = 3,
                 layers_lidar = 2,
                 steps = 3,
                 multiplier = 3,
                 search_mode = 2,
                 score_arch='branch_cls',
                 softmax_mode='dual_add',
                 test_mode=0,
                 affinity_op='minus_abs',
                 end_arch='v2',
                 end_mode='avg',
                 neg_threshold=0,
                 criterion = None):
        super(TrackingNetwork, self).__init__()
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.det_type = '3D'
        self.track_feats = None
        self._C_image = C_image
        self._C_lidar = C_lidar
        self._out_channels = out_channels
        self._layers_image = layers_image
        self._layers_lidar = layers_lidar
        self._steps = steps
        self._multiplier = multiplier
        self._criterion = criterion

        self.seq_len = 1
        self.score_arch = score_arch
        self.neg_threshold = neg_threshold
        self.test_mode = test_mode  # 0:image;1:image;2:fusion

        self.n_views = 1

        # build new end indicator
        if end_arch in ['v1', 'v2']:
            new_end = partial(eval("NewEndIndicator_%s" % end_arch),
                              kernel_size=5,
                              reduction=4,
                              mode=end_mode)
        in_channels = 512
        # build affinity matrix module
        assert in_channels != 0
        self.w_link = affinity_module(in_channels,
                                      new_end=new_end,
                                      affinity_op=affinity_op)

        # build negative rejection module
        if score_arch in ['branch_cls', 'branch_reg']:
            self.w_det = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1, 1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels // 2, 1, 1),
                nn.BatchNorm1d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels // 2, 1, 1, 1),
            )
        else:
            print("Not implement yet")

        self.softmax_mode = softmax_mode
        self.search_mode = search_mode

        # self.config = GlobalConfig()
        # self.encoder = Encoder(self.config)
        
        if search_mode == 2:
            self.stem_image = nn.Sequential(
                nn.Conv2d(3, C_image, kernel_size = 7, stride = 2, padding = 3, bias=False),
                nn.BatchNorm2d(C_image, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

            self.stem_lidar = nn.Sequential(
                nn.Conv2d(1, C_lidar, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(C_lidar, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

            C_prev_prev_i, C_prev_i, C_curr_i = C_image, C_image, C_image
            self.cells_image = nn.ModuleList()
            reduction_prev = False       
            for i in range(self._layers_image):
                if i in [self._layers_image // 3, 2 * self._layers_image // 3]:
                    C_curr_i *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev_i, C_prev_i, C_curr_i, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells_image += [cell]
                C_prev_prev_i, C_prev_i = C_prev_i, multiplier * C_curr_i
                
            self.image_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.image_fc = nn.Linear(C_prev_i, out_channels)

            C_prev_prev_l, C_prev_l, C_curr_l = C_lidar, C_lidar, C_lidar
            self.cells_lidar = nn.ModuleList()
            reduction_prev = False
            for i in range(self._layers_lidar):
                if i in [self._layers_lidar // 3, 2 * self._layers_lidar // 3]:
                    C_curr_l *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev_l, C_prev_l, C_curr_l, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells_lidar += [cell]
                C_prev_prev_l, C_prev_l = C_prev_l, multiplier * C_curr_l
            
            self.lidar_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.lidar_fc = nn.Linear(C_prev_l, out_channels)
        elif search_mode == 1:
            self.image_encoder = ImageCNN(512, normalize=True)
            self.stem_lidar = nn.Sequential(
                nn.Conv2d(1, C_lidar, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(C_lidar, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

            C_prev_prev_l, C_prev_l, C_curr_l = C_lidar, C_lidar, C_lidar
            self.cells_lidar = nn.ModuleList()
            reduction_prev = False
            for i in range(self._layers_lidar):
                if i in [self._layers_lidar // 3, 2 * self._layers_lidar // 3]:
                    C_curr_l *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev_l, C_prev_l, C_curr_l, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells_lidar += [cell]
                C_prev_prev_l, C_prev_l = C_prev_l, multiplier * C_curr_l
            
            self.lidar_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.lidar_fc = nn.Linear(C_prev_l, out_channels)
        elif search_mode == 0:
            self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)
            self.stem_image = nn.Sequential(
                nn.Conv2d(3, C_image, kernel_size = 7, stride = 2, padding = 3, bias=False),
                nn.BatchNorm2d(C_image, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
            C_prev_prev_i, C_prev_i, C_curr_i = C_image, C_image, C_image
            self.cells_image = nn.ModuleList()
            reduction_prev = False       
            for i in range(self._layers_image):
                if i in [self._layers_image // 3, 2 * self._layers_image // 3]:
                    C_curr_i *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev_i, C_prev_i, C_curr_i, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells_image += [cell]
                C_prev_prev_i, C_prev_i = C_prev_i, multiplier * C_curr_i
                
            self.image_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.image_fc = nn.Linear(C_prev_i, out_channels)


        self._initialize_alphas()

    def associate(self, objs, dets):
        link_mat, new_score, end_score = self.w_link(objs, dets)

        if self.softmax_mode == 'single':
            link_score = F.softmax(link_mat, dim=-1)
        elif self.softmax_mode == 'dual':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = link_score_prev.mul(link_score_next)
        elif self.softmax_mode == 'dual_add':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = (link_score_prev + link_score_next) / 2
        elif self.softmax_mode == 'dual_max':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = torch.max(link_score_prev, link_score_next)
        else:
            link_score = link_mat

        return link_score, new_score, end_score

    def feature(self, image_list, det_info):
        trans = None
        total_latency = 0.0
        lidar_list = det_info['points_bev']
        lidar_list = lidar_list.squeeze(0)
        # print(lidar_list.shape)
        bz, _, h, w = lidar_list.shape
        #print(image_list.shape)
        length, img_channel, _, _ = image_list.shape
        lidar_channel = lidar_list[0].shape[0]
        # print(h, w)
        
             
        image_tensor = image_list.view(bz * self.n_views * self.seq_len, img_channel, h, w)   # b, 3, h, w
        lidar_tensor = lidar_list.view(bz * self.seq_len, lidar_channel, h, w)   # b, 1, h, w

        if self.search_mode == 0:
            lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
            lidar_features = self.lidar_encoder._model.bn1(lidar_features)
            lidar_features = self.lidar_encoder._model.relu(lidar_features)
            lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
            lidar_features = self.lidar_encoder._model.layer1(lidar_features)
            lidar_features = self.lidar_encoder._model.layer2(lidar_features)
            lidar_features = self.lidar_encoder._model.layer3(lidar_features)
            lidar_features = self.lidar_encoder._model.layer4(lidar_features)
            lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
            lidar_features = torch.flatten(lidar_features, 1)
            lidar_features = lidar_features.view(bz, self.seq_len, -1)
            
            s0 = s1 = self.stem_image(image_tensor)
            for i, cell in enumerate(self.cells_image):
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
                total_latency += cell.cell_lat
            image_features = self.image_global_pooling(s1)
            image_features = torch.flatten(image_features, 1)
            image_features = image_features.view(bz, self.n_views * self.seq_len, -1)
            image_features = self.image_fc(image_features)
           
        elif self.search_mode == 1:
            image_features = self.image_encoder.features.conv1(image_tensor)
            image_features = self.image_encoder.features.bn1(image_features)
            image_features = self.image_encoder.features.relu(image_features)
            image_features = self.image_encoder.features.maxpool(image_features)

            image_features = self.image_encoder.features.layer1(image_features)
            image_features = self.image_encoder.features.layer2(image_features)
            image_features = self.image_encoder.features.layer3(image_features)
            image_features = self.image_encoder.features.layer4(image_features)
            image_features = self.image_encoder.features.avgpool(image_features)
            image_features = torch.flatten(image_features, 1)
            image_features = image_features.view(bz, self.n_views * self.seq_len, -1)

            s0 = s1 = self.stem_lidar(lidar_tensor)
            for i, cell in enumerate(self.cells_lidar):   
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
                total_latency += cell.cell_lat
            lidar_features = self.lidar_global_pooling(s1)     
            lidar_features = torch.flatten(lidar_features, 1)
            lidar_features = lidar_features.view(bz, self.seq_len, -1)
            lidar_features = self.lidar_fc(lidar_features)                      
        
        elif self.search_mode == 2:
            s0 = s1 = self.stem_image(image_tensor)
            for i, cell in enumerate(self.cells_image):
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
                total_latency += cell.cell_lat
            image_features = self.image_global_pooling(s1)
            image_features = torch.flatten(image_features, 1)
            image_features = image_features.view(bz, self.n_views * self.seq_len, -1)
            image_features = self.image_fc(image_features)

            s0 = s1 = self.stem_lidar(lidar_tensor)
            for i, cell in enumerate(self.cells_lidar):
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
                total_latency += cell.cell_lat
            lidar_features = self.lidar_global_pooling(s1)     
            lidar_features = torch.flatten(lidar_features, 1)
            lidar_features = lidar_features.view(bz, self.seq_len, -1)
            lidar_features = self.lidar_fc(lidar_features)


        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.permute(1,0).contiguous()
        fused_features = fused_features.unsqueeze(0)
        return fused_features, trans, total_latency

    def determine_det(self, dets, feats):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        if not self.training:
            # add mask
            if 'cls' in self.score_arch:
                det_scores = det_scores.sigmoid()

            mask = det_scores.lt(self.neg_threshold)
            det_scores -= mask.float()
        return det_scores

    def clear_mem(self):
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.track_feats = None
        return

    def set_eval(self):
        self.train(False)
        self.clear_mem()
        return

    def set_train(self):
        self.train(True)
        self.clear_mem()
        return

    def new(self):
        model_new = TrackingNetwork(self._C_image, self._C_lidar, self._out_channels, self._layers_image, self._layers_lidar, self._steps, self._multiplier, self.search_mode, criterion = self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, dets, det_info, dets_split):
        feats, trans, latency = self.feature(dets, det_info)
        det_scores = self.determine_det(dets, feats)

        start = 0
        link_scores = []
        new_scores = []
        end_scores = []
        for i in range(len(dets_split) - 1):
            prev_end = start + dets_split[i].item()
            end = prev_end + dets_split[i + 1].item()
            link_score, new_score, end_score = self.associate(
                feats[:, :, start:prev_end], feats[:, :, prev_end:end])
            link_scores.append(link_score.squeeze(1))
            new_scores.append(new_score)
            end_scores.append(end_score)
            start = prev_end

        if not self.training:
            fake_new = det_scores.new_zeros(
                (det_scores.size(0), link_scores[0].size(-2)))
            fake_end = det_scores.new_zeros(
                (det_scores.size(0), link_scores[-1].size(-1)))
            new_scores = torch.cat([fake_new] + new_scores, dim=1)
            end_scores = torch.cat(end_scores + [fake_end], dim=1)
        else:
            new_scores = torch.cat(new_scores, dim=1)
            end_scores = torch.cat(end_scores, dim=1)
            
        # print("latency", latency)
        return det_scores, link_scores, new_scores, end_scores, trans, latency

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        # self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def _loss(self, det_img, det_info, det_id, det_cls, det_split):
        det_score, link_score, new_score, end_score, trans, latency = self(
            det_img, det_info, det_split)
        # generate gt_y
        gt_det, gt_link, gt_new, gt_end = self.generate_gt(
            det_score[0], det_cls, det_id, det_split)

        # calculate loss
        loss = self._criterion(det_split, gt_det, gt_link, gt_new, gt_end,
                              det_score, link_score, new_score, end_score,
                              trans)
                              
        # TODO: latency loss added
        loss += latency * 2
        return loss

    def generate_gt(self, det_score, det_cls, det_id, det_split):
        gt_det = det_score.new_zeros(det_score.size())
        gt_new = det_score.new_zeros(det_score.size())
        gt_end = det_score.new_zeros(det_score.size())
        gt_link = []
        det_start_idx = 0

        for i in range(len(det_split)):
            det_curr_num = det_split[i]  # current frame i has det_i detects
            if i != len(det_split) - 1:
                link_matrix = det_score.new_zeros(
                    (1, det_curr_num, det_split[i + 1]))
            # Assign the score, according to eq1
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # g_det
                if det_cls[i][0][j] == 1:
                    gt_det[curr_det_idx] = 1  # positive
                else:
                    continue

                # g_link for successor frame
                if i == len(det_split) - 1:
                    # end det at last frame
                    gt_end[curr_det_idx] = 1
                else:
                    matched = False
                    det_next_num = det_split[i + 1]
                    for k in range(det_next_num):
                        if det_id[i][0][j] == det_id[i + 1][0][k]:
                            link_matrix[0][j][k] = 1
                            matched = True
                            break
                    if not matched:
                        # no successor means an end det
                        gt_end[curr_det_idx] = 1

                if i == 0:
                    # new det at first frame
                    gt_new[curr_det_idx] = 1
                else:
                    # look prev
                    matched = False
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if det_id[i][0][j] == det_id[i - 1][0][k]:
                            # have been matched during search in
                            # previous frame, no need to assign
                            matched = True
                            break
                    if not matched:
                        gt_new[curr_det_idx] = 1

            det_start_idx += det_curr_num
            if i != len(det_split) - 1:
                gt_link.append(link_matrix)

        return gt_det, gt_link, gt_new, gt_end

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                       if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal,
                            normal_concat=concat,
                            reduce=gene_reduce,
                            reduce_concat=concat)
        return genotype

    def predict(self, det_imgs, det_info, dets, det_split):
        det_score, link_score, new_score, end_score, _, _ = self(
            det_imgs, det_info, det_split)

        assign_det, assign_link, assign_new, assign_end = ortools_solve(
            det_score[self.test_mode],
            [link_score[0][self.test_mode:self.test_mode + 1]],
            new_score[self.test_mode], end_score[self.test_mode], det_split)

        assign_id, assign_bbox = self.assign_det_id(assign_det, assign_link,
                                                    assign_new, assign_end,
                                                    det_split, dets)
        aligned_ids, aligned_dets, frame_start = self.align_id(
            assign_id, assign_bbox)

        return aligned_ids, aligned_dets, frame_start

    def mem_assign_det_id(self, feats, assign_det, assign_link, assign_new,
                          assign_end, det_split, dets):
        det_ids = []
        v, idx = torch.max(assign_link[0][0], dim=0)
        for i in range(idx.size(0)):
            if v[i] == 1:
                track_id = idx[i].item()
                det_ids.append(track_id)
                self.track_feats[track_id] = feats[i:i + 1]
            else:
                new_id = self.last_id + 1
                det_ids.append(new_id)
                self.last_id += 1
                self.track_feats.append(feats[i:i + 1])

        for k, v in dets[0].items():
            dets[0][k] = v.squeeze(0) if k != 'frame_idx' else v
        dets[0]['id'] = torch.Tensor(det_ids).long()
        self.frames_id.append(det_ids)
        self.frames_det += dets
        assert len(self.track_feats) == (self.last_id + 1)

        return det_ids, dets

    def align_id(self, dets_ids, dets_out):
        frame_start = 0
        if len(self.used_id) == 0:
            # Start of a sequence
            self.used_id += dets_ids
            self.frames_id += dets_ids
            self.frames_det += dets_out
            max_id = 0
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    continue
                max_id = np.maximum(np.max(dets_ids[i]), max_id)
            self.last_id = np.maximum(self.last_id, max_id)
            return dets_ids, dets_out, frame_start
        elif self.frames_det[-1]['frame_idx'] != dets_out[0]['frame_idx']:
            # in case the sequence is not continuous
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_offset = self.last_id + 1
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i] + id_offset
                max_id = np.maximum(np.max(new_id), max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] += id_offset
            aligned_dets += dets_out
            self.last_id = np.maximum(self.last_id, max_id)
            self.frames_id += aligned_ids
            self.frames_det += aligned_dets
            return aligned_ids, aligned_dets, frame_start
        else:
            # the first frame of current dets
            # and the last frame of last dets is the same
            frame_start = 1
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_pairs = {}
            """
            assert len(dets_ids[0])== len(self.frames_id[-1])
            """
            # Calculate Id pairs
            for i in range(len(dets_ids[0])):
                # Use minimum because because sometimes
                # they are not totally the same
                has_match = False
                for j in range(len(self.frames_id[-1])):
                    if ((self.det_type == '3D'
                         and torch.sum(dets_out[0]['location'][i] !=
                                       self.frames_det[-1]['location'][j]) == 0
                         and torch.sum(dets_out[0]['bbox'][i] !=
                                       self.frames_det[-1]['bbox'][j]) == 0)
                            or (self.det_type == '2D' and torch.sum(
                                dets_out[0]['bbox'][i] != self.frames_det[-1]
                                ['bbox'][j]) == 0)):  # noqa

                        id_pairs[dets_ids[0][i]] = self.frames_id[-1][j]
                        has_match = True
                        break
                if not has_match:
                    id_pairs[dets_ids[0][i]] = self.last_id + 1
                    self.last_id += 1
            if len([v for k, v in id_pairs.items()]) != len(
                    set([v for k, v in id_pairs.items()])):
                print("ID pairs has duplicates!!!")
                print(id_pairs)
                print(dets_ids)
                print(dets_out[0])
                print(self.frames_id[-1])
                print(self.frames_det[-1])

            for i in range(1, len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i].copy()
                for j in range(len(dets_ids[i])):
                    if dets_ids[i][j] in id_pairs.keys():
                        new_id[j] = id_pairs[dets_ids[i][j]]
                    else:
                        new_id[j] = self.last_id + 1
                        id_pairs[dets_ids[i][j]] = new_id[j]
                        self.last_id += 1
                if len(new_id) != len(
                        set(new_id)):  # check whether there is duplicate
                    print('have duplicates!!!')
                    print(id_pairs)
                    print(new_id)
                    print(dets_ids)
                    print(dets_out)
                    print(self.frames_id[-1])
                    print(self.frames_det[-1])
                    import pdb
                    pdb.set_trace()

                max_id = np.maximum(np.max(new_id), max_id)
                self.last_id = np.maximum(self.last_id, max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] = torch.Tensor(new_id).long()
            # TODO: This only support check for 2 frame case
            if dets_out[1]['id'].size(0) != 0:
                aligned_dets += dets_out[1:]
                self.frames_id += aligned_ids
                self.frames_det += aligned_dets
            return aligned_ids, aligned_dets, frame_start

    def assign_det_id(self, assign_det, assign_link, assign_new, assign_end,
                      det_split, dets):
        det_start_idx = 0
        det_ids = []
        already_used_id = []
        fake_ids = []
        dets_out = []

        for i in range(len(det_split)):
            frame_id = []
            det_curr_num = det_split[i].item()
            fake_id = []
            det_out = get_start_gt_anno()
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # check w_det
                if assign_det[curr_det_idx] != 1:
                    fake_id.append(-1)
                    continue
                else:
                    # det_out.append(dets[i][j])
                    det_out['name'].append(dets[i]['name'][:, j])
                    det_out['truncated'].append(dets[i]['truncated'][:, j])
                    det_out['occluded'].append(dets[i]['occluded'][:, j])
                    det_out['alpha'].append(dets[i]['alpha'][:, j])
                    det_out['bbox'].append(dets[i]['bbox'][:, j])
                    det_out['dimensions'].append(dets[i]['dimensions'][:, j])
                    det_out['location'].append(dets[i]['location'][:, j])
                    det_out['rotation_y'].append(dets[i]['rotation_y'][:, j])

                # w_det=1, check whether a new det
                if i == 0:
                    if len(already_used_id) == 0:
                        frame_id.append(0)
                        fake_id.append(0)
                        already_used_id.append(0)
                        det_out['id'].append(torch.Tensor([0]).long())
                    else:
                        new_id = already_used_id[-1] + 1
                        frame_id.append(new_id)
                        fake_id.append(new_id)
                        already_used_id.append(new_id)
                        det_out['id'].append(torch.Tensor([new_id]).long())
                    continue
                elif assign_new[curr_det_idx] == 1:
                    new_id = already_used_id[-1] + 1 if len(
                        already_used_id) > 0 else 0
                    frame_id.append(new_id)
                    fake_id.append(new_id)
                    already_used_id.append(new_id)
                    det_out['id'].append(torch.Tensor([new_id]).long())
                else:
                    # look prev
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if assign_link[i - 1][0][k][j] == 1:
                            prev_id = fake_ids[-1][k]
                            frame_id.append(prev_id)
                            fake_id.append(prev_id)
                            det_out['id'].append(
                                torch.Tensor([prev_id]).long())
                            break

            assert len(fake_id) == det_curr_num
            fake_ids.append(fake_id)
            det_ids.append(np.array(frame_id))
            for k, v in det_out.items():
                if len(det_out[k]) == 0:
                    det_out[k] = torch.Tensor([])
                else:
                    det_out[k] = torch.cat(v, dim=0)
            det_out['frame_idx'] = dets[i]['frame_idx']
            dets_out.append(det_out)
            det_start_idx += det_curr_num
        return det_ids, dets_out
