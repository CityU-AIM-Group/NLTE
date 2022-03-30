# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import torch.nn.functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from maskrcnn_benchmark.modeling.utils import cat

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.cls_loss_weight = cfg.MODEL.CLS_LOSS_WEIGHT


    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        
        if self.training:
            domain_masks = cat([proposal.get_field("domain_labels") for proposal in proposals], dim=0)
            labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            domain_masks = None
            labels = None
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x, domain_masks, labels)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, None, None

        loss_classifier, loss_box_reg, da_ins_labels, labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        # if self.training:
        #     with torch.no_grad():
        #         da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        # da_ins_feas = self.feature_extractor(features, da_proposals)
        # class_logits, box_regression = self.predictor(da_ins_feas)
        # _, _, da_ins_labels, labels = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )

        others = {}
        others["labels"] = labels
        others["class_logits"] = class_logits 
        others["domain_labels"] = da_ins_labels
        others["box_regression"] = box_regression

        return (
            x,
            proposals,
            dict(loss_classifier=self.cls_loss_weight * loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_labels,
            others
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
