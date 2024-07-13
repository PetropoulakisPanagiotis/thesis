import json

from scipy.io import loadmat
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torchvision import transforms

from dataloaders.dataloader_dataset import DatasetPreprocess, ToTensorCustom
from utils import DistributedSamplerNoEvenlyDivisible


class DataLoaderCustom(object):
    def __init__(self, args, mode):
        self.mapping = None
        self.num_instances = 65  # Max instances in one image
        self.semantic_classes = ('void', 'bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects',
                                     'picture', 'sofa', 'table', 'tv', 'wall', 'window')
        self.num_semantic_classes = len(self.semantic_classes)
        self.mapping = None
        
        if args.dataset != 'scannet' and args.dataset != 'nyu':
            raise ValueError('No support for the given dataset')

        if args.dataset == 'scannet':
            mapping_40_to_13 = loadmat(
                '/usr/stud/petp/storage/user/petp/datasets/bts/utils/class13Mapping.mat')['classMapping13'][0][0]
            self.mapping = np.concatenate([[0], mapping_40_to_13[0][0]])
        
        if mode == 'train':
            self.training_samples = DatasetPreprocess(
                args, mode, transform=preprocessing_transforms(mode, args.segmentation, self.mapping))

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size, shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads, pin_memory=True, sampler=self.train_sampler)

        elif 'eval' in mode:
            self.testing_samples = DatasetPreprocess(
                args, mode, transform=preprocessing_transforms(mode, args.segmentation, self.mapping))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None

            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1, pin_memory=True,
                                   sampler=self.eval_sampler)
        else:
            print('mode should be one of \'train, online_eval\'. Got {}'.format(mode))


def preprocessing_transforms(mode, segmentation, classes_mapping=None):
    return transforms.Compose([ToTensorCustom(mode=mode, segmentation=segmentation, classes_mapping=classes_mapping)])
