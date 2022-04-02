import torch
import torch.nn as nn
from .operations import *
from .utils import drop_path
from .transfuser import ImageCNN, LidarEncoder

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            #print(h1.shape)
            #print(h2.shape)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
    

class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr


        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers

        self.stem0 = nn.Sequential(
            nn.Conv2d(3,
                      C // 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

class NetworkImageCNN(nn.Module):
    def __init__(self, C_image = 64, out_channels = 512, layers = 5, genotype = None):
        super(NetworkImageCNN, self).__init__()
        self._layers = layers
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)

        self.stem = nn.Sequential(
            nn.Conv2d(3, C_image, kernel_size = 7, stride = 2, padding = 3, bias=False),
            nn.BatchNorm2d(C_image, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        
        C_prev_prev, C_prev, C_curr = C_image, C_image, C_image
        self.n_views = 1   
        self.seq_len = 1
        self.cells = nn.ModuleList()
        reduction_prev = False       
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            
        self.image_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.image_fc = nn.Linear(C_prev, out_channels)
        
    def forward(self, image_list, lidar_list):
        lidar_list = lidar_list.squeeze(0)
        # print(lidar_list.shape)
        bz, _, h, w = lidar_list.shape
        length, img_channel, _, _ = image_list.shape
        lidar_channel = lidar_list[0].shape[0]

        image_tensor = image_list.view(bz * self.n_views * self.seq_len, img_channel, h, w)   # b, 3, h, w
        lidar_tensor = lidar_list.view(bz * self.seq_len, lidar_channel, h, w)   # b, 1, h, w
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

        s0 = s1 = self.stem(image_tensor)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        image_features = self.image_global_pooling(s1)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.n_views * self.seq_len, -1)
        image_features = self.image_fc(image_features)
        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.permute(1,0).contiguous()
        fused_features = fused_features.unsqueeze(0)
        return fused_features        

class NetworkLidarEncoder(nn.Module):
    def __init__(self, C_image = 3, C_lidar = 64, out_channels = 512, layers = 18, genotype = None):
        self.layers = layers
        
        """
        first implementation of serach for ResNet18
        """
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_lidar, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(C_lidar, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        
        C_prev_prev, C_prev, C_curr = C_lidar, C_lidar, C_lidar
        self.image_encoder = ImageCNN(512, normalize=True)
        self.n_views = 1   
        self.seq_len = 1
        
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        self.lidar_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.lidar_fc = nn.Linear(C_prev, out_channels)

    def forward(self, image_list, lidar_list):
        lidar_list = lidar_list.squeeze(0)
        # print(lidar_list.shape)
        bz, _, h, w = lidar_list.shape
        length, img_channel, _, _ = image_list.shape
        lidar_channel = lidar_list[0].shape[0]
        # print(h, w)
        
             
        image_tensor = image_list.view(bz * self.n_views * self.seq_len, img_channel, h, w)   # b, 3, h, w
        lidar_tensor = lidar_list.view(bz * self.seq_len, lidar_channel, h, w)   # b, 1, h, w

        s0 = s1 = self.stem(lidar_tensor)
        #image_features = self.image_encoder(image_tensor)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        # lidar_features = self.global_pooling(s1)
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
        lidar_features = self.lidar_global_pooling(s1)
        
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.seq_len, -1)
        # print("lidar" ,lidar_features.shape)
        # print("image", image_features.shape)
        lidar_features = self.lidar_fc(lidar_features)
        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.permute(1,0).contiguous()
        fused_features = fused_features.unsqueeze(0)
        return fused_features

"""
class NetworkEncoder(nn.Module):
    def __init__(self, C_image = 3, C_lidar = 64, out_channels = 512, layers = 18, genotype = None):
        super(NetworkEncoder, self).__init__()
        self.layers = layers
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_lidar, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(C_lidar, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        
        C_prev_prev, C_prev, C_curr = C_lidar, C_lidar, C_lidar
        self.image_encoder = ImageCNN(512, normalize=True)
        self.n_views = 1   
        self.seq_len = 1
        
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        self.lidar_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.lidar_fc = nn.Linear(C_prev, out_channels)
        
    def forward(self, image_list, lidar_list):
        lidar_list = lidar_list.squeeze(0)
        # print(lidar_list.shape)
        bz, _, h, w = lidar_list.shape
        length, img_channel, _, _ = image_list.shape
        lidar_channel = lidar_list[0].shape[0]
        # print(h, w)
        
             
        image_tensor = image_list.view(bz * self.n_views * self.seq_len, img_channel, h, w)   # b, 3, h, w
        lidar_tensor = lidar_list.view(bz * self.seq_len, lidar_channel, h, w)   # b, 1, h, w

        s0 = s1 = self.stem(lidar_tensor)
        #image_features = self.image_encoder(image_tensor)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        # lidar_features = self.global_pooling(s1)
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
        lidar_features = self.lidar_global_pooling(s1)
        
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.seq_len, -1)
        # print("lidar" ,lidar_features.shape)
        # print("image", image_features.shape)
        lidar_features = self.lidar_fc(lidar_features)
        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.permute(1,0).contiguous()
        fused_features = fused_features.unsqueeze(0)
        return fused_features
""" 

class NetworkEncoder(nn.Module):
    def __init__(self, C_image = 64, C_lidar = 64, out_channels = 512, layers_image = 2, layers_lidar = 2,  genotype_image = None, genotype_lidar = None):
        super(NetworkEncoder, self).__init__()
        self.image_layers = layers_image
        self.lidar_layers = layers_lidar
        
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
        self.n_views = 1   
        self.seq_len = 1

        C_prev_prev_i, C_prev_i, C_curr_i = C_image, C_image, C_image
        self.cells_image = nn.ModuleList()
        reduction_prev = False       
        for i in range(self.image_layers):
            if i in [self.image_layers // 3, 2 * self.image_layers // 3]:
                C_curr_i *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype_image, C_prev_prev_i, C_prev_i, C_curr_i, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells_image += [cell]
            C_prev_prev_i, C_prev_i = C_prev_i, cell.multiplier * C_curr_i
            
        self.image_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.image_fc = nn.Linear(C_prev_i, out_channels)

        C_prev_prev_l, C_prev_l, C_curr_l = C_lidar, C_lidar, C_lidar
        self.cells_lidar = nn.ModuleList()
        reduction_prev = False
        for i in range(self.lidar_layers):
            if i in [self.lidar_layers // 3, 2 * self.lidar_layers // 3]:
                C_curr_l *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype_lidar, C_prev_prev_l, C_prev_l, C_curr_l, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells_lidar += [cell]
            C_prev_prev_l, C_prev_l = C_prev_l, cell.multiplier * C_curr_l
        
        self.lidar_global_pooling =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.lidar_fc = nn.Linear(C_prev_l, out_channels)
        
    def forward(self, image_list, lidar_list):
        lidar_list = lidar_list.squeeze(0)
        # print(lidar_list.shape)
        bz, _, h, w = lidar_list.shape
        length, img_channel, _, _ = image_list.shape
        lidar_channel = lidar_list[0].shape[0]
        # print(h, w)
        
             
        image_tensor = image_list.view(bz * self.n_views * self.seq_len, img_channel, h, w)   # b, 3, h, w
        lidar_tensor = lidar_list.view(bz * self.seq_len, lidar_channel, h, w)   # b, 1, h, w


        s0 = s1 = self.stem_image(image_tensor)
        for i, cell in enumerate(self.cells_image):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        image_features = self.image_global_pooling(s1)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.n_views * self.seq_len, -1)
        image_features = self.image_fc(image_features)

        s0 = s1 = self.stem_lidar(lidar_tensor)
        for i, cell in enumerate(self.cells_lidar):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        lidar_features = self.lidar_global_pooling(s1)     
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.seq_len, -1)
        lidar_features = self.lidar_fc(lidar_features)


        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.permute(1,0).contiguous()
        fused_features = fused_features.unsqueeze(0)
        return fused_features
