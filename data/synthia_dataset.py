import os
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from utils_ import get_label_mapper


class Synthia_loader(object):
    """
    Synthia    synthetic dataset
    for domain adaptation to Cityscapes
    """

    def __init__(self, opt, logger, augmentations=None):
        self.opt = opt
        self.logger = logger
        self.root = Path(opt.src_rootpath)
        self.augmentations = augmentations
        self.n_class = opt.n_class
        self.img_size = (760, 1280)     # [h, w]

        # Dataset directory
        self.image_base_path = self.root / "RGB"
        self.label_base_path = self.root / "GT/LABELS"

        # Get training file list
        self.ids = list(sorted(self.image_base_path.glob("*.png")))
        if len(self.ids) == 0:
            raise Exception(f"No files found in {self.image_base_path}")
        self.logger.info(f"Found {len(self.ids)} images")

        # Get label index mapper
        self.label2id = get_label_mapper("synthia", self.n_class)

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
        
        image = Image.open(image_path)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)[:, :, 2]

        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label = cv2.resize(label, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # map label indices to [0, 1, 2, ...]
        label = Image.fromarray(self.label2id[label])
        
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
