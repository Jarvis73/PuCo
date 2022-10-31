import torch.utils.data
from data.augmentations import Compose, RandomSized, RandomCrop, RandomHorizontallyFlip
from data import cityscapes_dataset, gta5_dataset, synthia_dataset

from data.DataProvider import DataProvider


def find_dataset_using_name(name):
    if name == "cityscapes":
        return cityscapes_dataset.Cityscapes_loader
    elif name == "gta5":
        return gta5_dataset.GTA5_loader
    elif name == 'synthia':
        return synthia_dataset.Synthia_loader
    else:
        raise ValueError


def create_dataset(opt, logger, **kwargs):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, logger, **kwargs)
    dataset = data_loader.load_data()
    return dataset


def get_composed_augmentations(opt):
    return Compose([RandomSized(opt.rng, opt.resize),
                    RandomCrop(opt.rcrop),
                    RandomHorizontallyFlip(opt.hflip)])


class CustomDatasetDataLoader(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger

        # -------------------------------------------------------
        #              Source training data loader               
        # -------------------------------------------------------
        source_cls = find_dataset_using_name(opt.src_dataset)
        data_aug = get_composed_augmentations(opt) if opt.aug else None
        self.source_train = source_cls(opt, logger, augmentations=data_aug)
        self.logger.info(f"dataset {self.source_train.__class__.__name__} for source was created")

        # infinity loader
        self.source_train_loader = DataProvider(self.source_train,
                                                batch_size=opt.bs,
                                                shuffle=opt.shuffle,
                                                num_workers=int(opt.num_workers),
                                                drop_last=opt.droplast,
                                                pin_memory=True,)

        # -------------------------------------------------------
        #              Target training data loader
        # -------------------------------------------------------
        target_cls = find_dataset_using_name(opt.tgt_dataset)
        data_aug = get_composed_augmentations(opt) if opt.aug else None
        self.target_train = target_cls(opt, logger, augmentations=data_aug, split='train')
        self.logger.info(f"dataset {self.target_train.__class__.__name__} for target was created")

        # epoch-wise loader
        self.target_train_loader = torch.utils.data.DataLoader(self.target_train,
                                                               batch_size=opt.bs,
                                                               shuffle=opt.shuffle,
                                                               num_workers=int(opt.num_workers),
                                                               drop_last=opt.droplast,
                                                               pin_memory=True)

        # -------------------------------------------------------
        #               Target validation data loader
        # -------------------------------------------------------
        self.target_valid = target_cls(opt, logger, augmentations=None, split='val')
        logger.info(f"dataset {self.target_valid.__class__.__name__} for target validation has been created")

        # epoch-wise loader
        self.target_valid_loader = torch.utils.data.DataLoader(self.target_valid,
                                                               batch_size=opt.bs,
                                                               shuffle=False,
                                                               num_workers=int(opt.num_workers),
                                                               drop_last=False,
                                                               pin_memory=True)

        # ----------------------------------------------------------------------
        #              Target pseudo-label-generation data loader
        # ----------------------------------------------------------------------
        self._target_pl = None
        self._target_pl_loader = None

    def load_data(self):
        return self

    def _create_pl_loader(self):
        opt = self.opt
        target_cls = find_dataset_using_name(opt.tgt_dataset)
        self._target_pl = target_cls(self.opt, self.logger, augmentations=None, split='train')
        self._target_pl_loader = torch.utils.data.DataLoader(self._target_pl,
                                                            batch_size=opt.bs,
                                                            shuffle=False,
                                                            num_workers=int(opt.num_workers),
                                                            drop_last=False,
                                                            pin_memory=True)

    @property
    def target_pl(self):
        if self._target_pl is None:
            self._create_pl_loader()
        return self._target_pl
    
    @property
    def target_pl_loader(self):
        if self._target_pl_loader is None:
            self._create_pl_loader()
        return self._target_pl_loader
