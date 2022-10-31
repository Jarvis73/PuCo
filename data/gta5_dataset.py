import os
import torch
import numpy as np
from PIL import Image
import scipy.io as io
from pathlib import Path

from utils_ import get_label_mapper


class GTA5_loader(object):
    """
    GTA5    synthetic dataset
    for domain adaptation to Cityscapes
    """

    def __init__(self, opt, logger, augmentations=None):
        self.opt = opt
        self.logger = logger
        self.root = Path(opt.src_rootpath)
        self.augmentations = augmentations
        self.n_class = opt.n_class
        self.img_size = (1052, 1914)    # [h, w]

        # Dataset directory
        self.image_base_path = self.root / "images"
        self.label_base_path = self.root / "labels"
        
        # Get training file list
        splits = io.loadmat(self.root / "split.mat")
        ids = np.concatenate((splits['trainIds'][:,0], splits['valIds'][:,0], splits['testIds'][:,0]))
        self.ids = [self.label_base_path / f"{i + 1:05d}.png" for i in range(len(ids))]
        if len(self.ids) == 0:
            raise Exception(f"No files found in {self.image_base_path}")
        self.logger.info(f"Found {len(self.ids)} images")

        # Get label index mapper
        self.label2id = get_label_mapper("gta5", self.n_class)        

        self.class_names = {
            19: "road sidewalk building wall fence pole traffic_light traffic_sign vegetation terrain "\
                "sky person rider car truck bus train motorcycle bicycle".split(),
            16: "road sidewalk building wall fence pole traffic_light traffic_sign "\
                "vegetation sky person rider car bus motorcycle bicycle".split()
        }
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        label_path = self.ids[index]
        image_path = self.image_base_path / label_path.name
        
        # Load image and label
        image = Image.open(image_path)
        label = Image.open(label_path)
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Map label indices to [0, 1, 2, ...]
        label = Image.fromarray(self.label2id[np.array(label)])

        # Data preprocessing
        if self.augmentations is not None:
            image, [label], _ = self.augmentations(image, [label])
        image, label = self.transform(image, label)

        input_dict = {
            'image': image,
            'label': label,
            'image_path': str(self.ids[index])
        }
        
        return input_dict

    def transform(self, image, label):
        image = np.array(image, np.float32)
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        label = np.array(label, np.uint8)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label

    def cname(self, idx):
        return self.class_names[self.n_class][idx]
