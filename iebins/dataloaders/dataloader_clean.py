import json

from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torchvision import transforms

from dataloaders.dataloader_dataset import DatasetPreprocess, ToTensorCustom 
from utils_clean import DistributedSamplerNoEvenlyDivisible


class DataLoaderCustom(object):
    def __init__(self, args, mode):
        if args.dataset == 'nyu':
            self.semantic_classes = json.load(open(args.data_path.split('train')[0] + 'labels.json', 'r'))
            self.num_semantic_classes = len(self.semantic_classes)
            self.num_instances = 63 # Max instances in one image

        if args.dataset == 'scannet':
            self.semantic_classes = ('void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
            self.num_semantic_classes = 21
            self.num_instances = 20

        if mode == 'train':
            self.training_samples = DatasetPreprocess(args, mode, transform=preprocessing_transforms(mode, args.segmentation))

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            
            self.testing_samples = DatasetPreprocess(args, mode, transform=preprocessing_transforms(mode, args.segmentation))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DatasetPreprocess(args, mode, transform=preprocessing_transforms(mode, args.segmentation))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def preprocessing_transforms(mode, segmentation):
    return transforms.Compose([
        ToTensorCustom(mode=mode, segmentation=segmentation)
    ])
