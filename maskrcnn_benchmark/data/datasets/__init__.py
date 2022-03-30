# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .clipart import ClipartDataset
from .concat_dataset import ConcatDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "ClipartDataset"]
