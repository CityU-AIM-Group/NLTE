# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
from posixpath import join
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer
from maskrcnn_benchmark.layers import SCELoss
from copy import deepcopy

from .loss import make_ca_da_heads_loss_evaluator
from maskrcnn_benchmark.modeling.utils import cat


class CADAHead(nn.Module):

    def __init__(self, in_channels, weight):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CADAHead, self).__init__()
        self.grl = GradientScalarLayer(-1.0*weight)
        self.ca_dis = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.ReLU(),
        )

        self.domain_classifier = nn.Linear(1024,1)
        
        self.sigmoid = nn.Sigmoid()
        
        for l in self.ca_dis:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, std=0.01)
                nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.domain_classifier.weight, std=0.05)
        nn.init.constant_(self.domain_classifier.bias, 0)

    def forward(self, x):
        x = self.grl(x)
        x = self.ca_dis(x)
        x = self.domain_classifier(x)
        x = self.sigmoid(x)
        return x

class InformationPrototype(nn.Module):
    def __init__(self, config, in_channels, num_classes, threshold=0.1):
        super(InformationPrototype, self).__init__()

        self.cfg = config
        # self.prototypes = torch.autograd.Variable(torch.zeros(num_classes, 2048), requires_grad=False)
        self.register_buffer('prototypes', torch.zeros(num_classes, 2048))
        self.register_buffer('step', torch.tensor([0]*num_classes))
        self.register_buffer('thresholds', torch.tensor([threshold] * num_classes))

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7) # 256 2048 1 1

    def update_prototype(self, x_new, max_logit, max_cls, prototypes, momentum=0.5):

        '''
            x_new: 256, 1024
            max_logit: 256
            max_cls: 256
            prototypes: C * 1024
        '''
        if self.cfg.MODEL.MY_HEAD_UPDATE == 'mean':
            for idx, cls in enumerate(list(max_cls)):
                if max_logit[idx] < self.thresholds[max_cls[idx]]:
                    prototypes = prototypes
                else:
                    self.step[cls] += 1
                    # prototype[cls] = prototype[cls] * momentum + x[idx] * (1 - momentum)
                    prototypes[cls] += x_new[idx]
                    prototypes[cls] /= self.step[cls]
                    # update cls-wise threshold
                    self.thresholds[cls] = self.thresholds[cls] * momentum + max_logit[idx] * (1 - momentum)
                    # print(self.thresholds)
                    
        elif self.cfg.MODEL.MY_HEAD_UPDATE == 'cos':
            C, D = prototypes.shape
            local_ptyps = torch.zeros_like(prototypes)
            local_ptyps_cls_count = torch.zeros(C).to(local_ptyps.device)
            for idx, cls in enumerate(list(max_cls)):
                local_ptyps[cls] += x_new[idx]
                local_ptyps_cls_count[cls] += 1

            exist_indx = local_ptyps_cls_count.type(torch.bool)

            local_ptyps[exist_indx] = local_ptyps[exist_indx] / (local_ptyps_cls_count[exist_indx].unsqueeze(1).expand(-1, D))

            momentum = torch.cosine_similarity(prototypes[exist_indx], local_ptyps[exist_indx]).unsqueeze(1)
            prototypes[exist_indx] = prototypes[exist_indx] * momentum + local_ptyps[exist_indx] * (1 - momentum)

        else:
            pass


        return prototypes

    def forward(self, x, class_logits):
        '''
            x : 256, 1024, 7, 7
            class_logits : 256, 21
        '''
        class_logits_new = class_logits.clone().detach()
        pred_normed = F.softmax(class_logits_new, dim=1)
        max_logit = pred_normed.max(dim=1)[0] # 256
        max_cls = pred_normed.argmax(dim=1) # 256
        self.prototypes = self.prototypes.to(x.device)

        x_mapped = self.avgpool(x).squeeze()
        x_new = x_mapped.clone().detach()
        # x_new = self.avgpool(x_new).squeeze()

        self.prototypes = self.update_prototype(x_new, max_logit, max_cls, self.prototypes)

        prototypes = self.prototypes
        step = self.step
        return prototypes, step, x, class_logits, max_cls, x_mapped


class CategoryAwareDAHEAD(torch.nn.Module):

    def __init__(self, cfg):
        super(CategoryAwareDAHEAD, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor

        self.loss_evaluator = make_ca_da_heads_loss_evaluator(self.cfg)

        self.grl_tn = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.source_prototype_buffer = InformationPrototype(cfg.clone(), num_ins_inputs, num_classes=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
        self.target_prototype_buffer = InformationPrototype(cfg.clone(), num_ins_inputs, num_classes=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

        self.ca_da_head = CADAHead(in_channels = num_ins_inputs + self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                                    weight=self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.module_avgpool = nn.AvgPool2d(kernel_size=7, stride=7) # 256 2048 1 1

        self.mgrm_on = cfg.MODEL.NOISETRANSFER_DA_ON
        self.ea_on = cfg.MODEL.NOISEGRADMATCH_ON
        self.loss_transition_matrix_weight = cfg.MODEL.MGRM_WEIGHT
        self.mgrm_loss_func = cfg.MODEL.MGRM_LOSS_FUNC


    def forward(self, ins_features, others):

        if self.training:
            losses = {}
            # print(ins_features.shape) # 512, 2048, 7, 7
            labels = others["labels"] # 256
            class_logits = others["class_logits"] # 512, 21
            box_regression = others["box_regression"] # 512, 84
            domain_labels = others["domain_labels"] # 512, [1 * 256, 0 * 256]

            
            if self.mgrm_on:
                source_class_logits = class_logits[domain_labels, :] # [256, 21]
                source_feature = ins_features[domain_labels, :, :, :]
                target_class_logits = class_logits[1 - domain_labels, :] # [256, 21]
                target_feature = ins_features[1 - domain_labels, :, :, :]

                # 256, 1024, 7, 7 | 256, 21 ==> num_classes * 1024 | []*21 | [256, 1024, 7, 7] | [256, 21] | [256, 1]
                source_prototypes, s_step, source_feature, source_class_logits, max_source_cls, source_feature_mapped = self.source_prototype_buffer(source_feature, source_class_logits) # num_classes * 1024
                target_prototypes, t_step, target_feature, target_class_logits, max_target_cls, target_feature_mapped = self.target_prototype_buffer(target_feature, target_class_logits) # num_classes * 1024

                # generate prototype transition matrix
                C, D = source_prototypes.shape
                f1 = source_prototypes.unsqueeze(0).expand(C, C, D)
                f2 = target_prototypes.unsqueeze(1).expand(C, C, D)
                ptm = F.cosine_similarity(f1, f2, dim=2)[1:, 1:] # C, C

                # generate batch-wise label and target prototype transition matrix
                bwl_ptyps = torch.zeros_like(source_prototypes)
                bwl_ptyps_cls_count = torch.zeros(C).to(bwl_ptyps.device)
                for idx, cls in enumerate(list(labels)):                    
                    if cls == -10:
                        source_feature_mapped_cur = source_feature_mapped[idx].repeat(bwl_ptyps.size()[0], 1)
                        source_feature_mapped_cur = F.cosine_similarity(bwl_ptyps, source_feature_mapped_cur)
                        cls = torch.argmax(source_feature_mapped_cur)
                    bwl_ptyps[cls] += source_feature_mapped[idx]
                    bwl_ptyps_cls_count[cls] += 1
                exist_indx = bwl_ptyps_cls_count.type(torch.bool)
                bwl_ptyps[exist_indx] = bwl_ptyps[exist_indx] / (bwl_ptyps_cls_count[exist_indx].unsqueeze(1).expand(-1, D))
                bwl_ptyps = bwl_ptyps.unsqueeze(0).expand(C, C, D)
                bwltpm = F.cosine_similarity(bwl_ptyps, f2, dim=2)[1:, 1:] # C, C

                if self.mgrm_loss_func == 'l1':
                    losses["loss_relation_matrix"] = self.loss_transition_matrix_weight * F.l1_loss(bwltpm[exist_indx[1:]], ptm[exist_indx[1:]])
                elif self.mgrm_loss_func == 'l2':
                    losses["loss_relation_matrix"] = self.loss_transition_matrix_weight * F.mse_loss(bwltpm[exist_indx[1:]], ptm[exist_indx[1:]])
                else:
                    raise NotImplementedError

            if self.ea_on:
                # BOX ALIGNMENT
                feat_pooled = self.module_avgpool(ins_features).squeeze()
                op_out = cat([feat_pooled, class_logits], dim=1)
                ca_da_pred = self.ca_da_head(op_out)

                losses["loss_ea"] = F.binary_cross_entropy_with_logits(
                                        torch.squeeze(ca_da_pred), domain_labels.type(torch.cuda.FloatTensor)
                                    )

            return losses

        return {}

def build_ca_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return CategoryAwareDAHEAD(cfg)
    return []
