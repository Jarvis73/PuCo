from pathlib import Path
import numpy as np

import torch

from data.augmentations import *
from utils_ import get_label_mapper


class Cityscapes_loader(object):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    def __init__(self, opt, logger, augmentations=None, split='train'):
        self.opt = opt
        self.logger = logger
        self.root = Path(opt.tgt_rootpath)
        self.split = split
        self.augmentations = augmentations
        self.n_class = opt.n_class
        self.img_size = (1024, 2048)    # [h, w]

        # Dataset directory
        self.images_base = self.root / "leftImg8bit" / self.split
        self.labels_base = self.root / "gtFine" / self.split
        self.pseudo_label_base = None

        # Get training file list
        self.files = sorted(list(self.images_base.glob("*/*_leftImg8bit.png")))
        if not self.files:
            raise Exception(f"No files for split=[{self.split}] found in {self.images_base}")
        self.logger.info(f"Found {len(self.files)} {self.split} images")

        # Get label index mapper
        self.label2id = get_label_mapper("cityscapes", self.n_class)

        # Photographic augmentation for intra-domain consistency
        if opt.intra_consist:
            self.photo_aug = Compose([
                *((RandomGaussianBlur(p=opt.blur),) if opt.blur > 0 else ()),    # removing it is better
                *((RandomShift(p=opt.shift),) if opt.shift else ()),
                *((RandomHorizontallyFlip(p=0.5),) if opt.flip else ()),
                RandomColorJitter(jitter=0.4, p=0.5),
                RandomGreyscale(p=0.2),
            ])

        self.class_names = {
            19: "road sidewalk building wall fence pole traffic_light traffic_sign vegetation terrain "\
                "sky person rider car truck bus train motorcycle bicycle".split(),
            16: "road sidewalk building wall fence pole traffic_light traffic_sign "\
                "vegetation sky person rider car bus motorcycle bicycle".split()
        }
    
    def __len__(self):
        if self.split == 'train':
            return self.opt.train_n or len(self.files)
        else:
            return self.opt.test_n or len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        label_path = self.labels_base / image_path.parts[-2] / (image_path.name[:-15] + "gtFine_labelIds.png")

        # Load image and label
        image = Image.open(image_path)  # [1024, 2048, 3]
        label = Image.open(label_path)  # [1024, 2048]

        # Map label indices to [0, 1, 2, ...]
        label = Image.fromarray(self.label2id[np.array(label)])

        # Load pseudo label
        pseudo_label = None
        if self.split == 'train' and self.pseudo_label_base:
            pseudo_label_path = self.pseudo_label_base / image_path.name
            pseudo_label = Image.open(pseudo_label_path)

        # Data preprocessing
        kwargs1, kwargs2 = None, None
        image2 = None     # the second view of intra-domain consistency
        if self.augmentations is not None:
            image, (label, pseudo_label), _ = self.augmentations(image, [label, pseudo_label])
            if self.opt.intra_consist:
                image2, _, kwargs2 = self.photo_aug(image, [])
                image, _, kwargs1 = self.photo_aug(image, [])
        (image, image2), (label, pseudo_label) = self.transform([image, image2], [label, pseudo_label])

        input_dict = {
            'image': image,
            'label': label,
            'image2': image2,
            'pseudo_label': pseudo_label,
            'image_path': str(self.files[index]),
            'kwargs1': kwargs1,
            'kwargs2': kwargs2,
        }
        
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        return input_dict

    def transform(self, images, labels):
        for i in range(len(images)):
            if images[i] is not None:
                images[i] = np.array(images[i], np.float32) / 255.0
                images[i] = torch.from_numpy(images[i].transpose(2, 0, 1)).float()

        for i in range(len(labels)):
            if labels[i] is not None:
                labels[i] = torch.from_numpy(np.array(labels[i], np.uint8)).long()

        return images, labels

    def cname(self, idx):
        return self.class_names[self.n_class][idx]
