from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import affinity_module
from .new_end import *  # noqa
from .score_net import *  # noqa
from .transfuser import GlobalConfig, Encoder
from .res_model import NetworkEncoder as Network
from .genotypes import *

class TrackingNet(nn.Module):
    def __init__(self,
                 seq_len,
                 score_arch='cls',
                 softmax_mode='single',
                 test_mode=0,
                 affinity_op='multiply',
                 end_arch='v2',
                 end_mode='avg',
                 without_reflectivity=True,
                 neg_threshold=0):
        super(TrackingNet, self).__init__()
        self.seq_len = seq_len
        self.score_arch = score_arch
        self.neg_threshold = neg_threshold
        self.test_mode = test_mode  # 0:image;1:image;2:fusion
        point_in_channels = 4 - int(without_reflectivity)

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

        # self.config = GlobalConfig()
        # self.encoder = Encoder(self.config)
        
        genotype = low
        self.encoder = Network(search_mode = 2, genotype_image = genotype, genotype_lidar = genotype)


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

    def feature(self, dets, det_info):
        trans = None
        points = det_info['points_bev']
        feats = self.encoder(dets, points)
        return feats, trans

    def determine_det(self, dets, feats):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        if not self.training:
            # add mask
            if 'cls' in self.score_arch:
                det_scores = det_scores.sigmoid()

            mask = det_scores.lt(self.neg_threshold)
            det_scores -= mask.float()
        return det_scores

    def forward(self, dets, det_info, dets_split):
        feats, trans = self.feature(dets, det_info)
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
        return det_scores, link_scores, new_scores, end_scores, trans
