# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn
import torch.nn.functional as F

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        self.source_inner = nn.Linear(2048, 2048)
        self.target_inner = nn.Linear(2048, 2048)
        self.relation = nn.Linear(2048, 2048)
        for layer in [self.source_inner, self.target_inner, self.relation]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def graph(self, feat):
        N, D = feat.shape
        f1 = feat.unsqueeze(0).expand(N, N, D)
        f2 = feat.unsqueeze(1).expand(N, N, D)
        mat = F.cosine_similarity(f1, f2, dim=2)
        del f1, f2
        return mat

    def forward(self, x, domain_masks, labels):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 512, 1024, 7, 7 ==> 512, 1024
        # graph_feat
        x_clone = x.clone().detach()

        if self.training: # during inference, the domain is target (0)
            source = x_clone[domain_masks, :]
            source_mat = self.graph(source)
            x[domain_masks, :] = self.source_inner(torch.mm(source_mat, x[domain_masks, :]).softmax(dim=-1)) + x[domain_masks, :]
            del source_mat

            target = x_clone[1 - domain_masks, :]
            target_mat = self.graph(target)
            x[1 - domain_masks, :] = self.target_inner(torch.mm(target_mat, x[1 - domain_masks, :]).softmax(dim=-1)) + x[1 - domain_masks, :]
            del target_mat
        else:
            target = x_clone
            target_mat = self.graph(target)
            x = self.target_inner(torch.mm(target_mat, x).softmax(dim=-1)) + x
            del target_mat

        x_clone = x.clone().detach()
        mat = self.graph(x_clone)
        x = self.relation(torch.mm(mat, x).softmax(dim=-1)) + x

        del x_clone
        del mat
        cls_logit = self.cls_score(x) # N*C

        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
