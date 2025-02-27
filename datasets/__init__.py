# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .catheter import build as build_catheter
from .face import build as build_face


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


# Here we build the coco dataset
def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'catheter':
        return build_catheter(image_set, args)
    if args.dataset_file == 'face':
        return build_face(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
