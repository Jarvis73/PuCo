import os
import sys
import random
import shutil
import tarfile
from pathlib import Path

import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import numpy as np
from sacred import SETTINGS, Experiment
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from constants import on_cloud
from loggers import get_global_logger

SETTINGS.DISCOVER_SOURCES = "sys"
SETTINGS.DISCOVER_DEPENDENCIES = "sys"

### Define experiment paths ###
CLOUD_DIR = Path("afs/output/models_puco")
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "datasets"                                 # storing datasets
PRE_DIR = CLOUD_DIR if on_cloud else PROJECT_DIR / "pretrained"     # storing pretrained models
LOG_DIR = PROJECT_DIR / "output"                                    # storing outputs for all the experiments
PL_DIR = PROJECT_DIR / "Pseudo"                                     # storing pseudo labels
STATUS_DIR = PL_DIR / "status"                                      # storing CBST thresholds


def settings(ex: Experiment):
    # Track outputs in sacred
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # ============================================================
    #    Configuration list
    # ============================================================
    @ex.config
    def configurations():
        # Global configurations
        model_name = "deeplabv2"        # deeplabv2, fixed
        task_name = "gta2city"          # UDA task name, gta2city|syn2city
        log_dir = str(LOG_DIR)          # directory to save experiment outputs
        pl_dir = str(PL_DIR)            # directory to save all the pseudo labels
        status_dir = str(STATUS_DIR)    # directory to save CBST thresholds

        print_interval = 50             # print interval, by iterations
        val_interval = 1000             # validation interval, by iterations

        mongo_host = ""                 # for MongoObserver
        mongo_port = 8854               # for MongoObserver
        fileStorage = True              # use fileStorageObserver
        mongodb = False                 # use MongoObserver

        # Training configurations
        seed = 1337                     # random seed
        stage = "self_training"         # source_only|target_only|warm_up|self_training
        lr = 0.0008                     # learning rate
        bs = 8                          # batch size
        epochs = 720                    # training epochs
        nstages = 3                     # number of self-training stages
        stop_stage = 0                  # stop after stage `stop`, 1|2|3
        amp = False                     # using automatic mixed precision

        momentum = 0.9                  # momentum of SGD optimizer
        nesterov = False                # nesterov of SGD optimizer
        weight_decay = 2e-4             # weight decay of SGD optimizer

        # Model configurations
        os = 8                          # Deeplabv2 output stride
        freeze_bn = False               # freeze BN layers and set to eval mode
        droprate = 0.1                  # drop rate of additional dropout layers
        drop_layers = (4,)              # dropout layers
        clf_droprate = 0.1              # classifier drop rate
        droppath = 0                    # drop rate of drop path in drop_layers
        strict = False                  # strictly loading weights

        # Data configuration
        src_dataset = "gta5"                            # gta5|synthia
        tgt_dataset = "cityscapes"                      # cityscapes
        src_rootpath = str(DATA_DIR / "GTA5")           # source domain data path
        tgt_rootpath = str(DATA_DIR / "CityScape")      # target domain data path
        train_n = 0                                     # number of training samples
        test_n = 0                                      # number of testing samples

        use_pseudo_label = True                         # use pseudo labels in experiments

        rng = (0.5, 1.5)                # random resize times
        resize = 2200                   # resize long size
        rcrop = (896, 512)              # rondom crop size
        hflip = 0.5                     # random flip probility
        # Intra-Domain Views
        blur = 0                        # random gaussian blur
        shift = 0                       # random shift
        flip = False                    # random flip

        n_class = 19                    # number of classes, 19|16
        num_workers = 20                # number of workers
        droplast = True                 # drop last samples
        shuffle = True                  # shuffle data
        aug = True                      # use data augmentation

        # Checkpoints
        init_path = str(PRE_DIR / 'resnet101-5d3b4d8f.pth')         # imagenet pretrained model
        warmup_expid = f'warmup_{src_dataset}'
        warmup_path = str(PRE_DIR / f'{warmup_expid}/best_model.pth')   # warmup model
        resume_path = None
        save_all = True

        # -------------------------------------------------------------------------------------------
        # Pseudo lables
        thresh = [0.2, 0.4, 0.4]                # CBST thresholds in n stages
        ds_rate = 4

        # Warmup losses
        adv = 0.01                              # weight of adversarial loss

        # Self-training losses
        sce = True                              # use symmetry cross entropy loss
        sce_alpha = 0.1                         # weight for symmetry cross entropy loss
        sce_beta = 1.0                          # weight for symmetry cross entropy loss
        reset_all = True                        # reset momentum and learning rate
        use_best = False                        # use best model when starting a new epoch

        # Cross-domain consistency losses
        cross_consist = True                    # use cross-domain consistency
        style_transfer_layers = [1, 2]          # style transfer layers
        style_transfer_sg = True                # stop gradient at the start of the new branch
        stat_dict_size = bs                     # size of the statistics directory
        srcW = 2.0                              # weight for cross-domain consistency loss

        # Intra-domain consistency losses
        stu_tea = True                          # use student-teacher constraint
        stu_stu = True                          # use student-student constraint
        intra_consist = stu_tea or stu_stu      # use intra-domain consistency, two views are processed
        proj_lr_const = True                    # projector using constant learning rate (SimSiam)
        q_momuntum = 0.999                      # momemtum of EMA model
        stu_teaW = 1.0                          # weight for student-teacher loss
        stu_stuW = 5.0                          # weight for student-student loss
        inc_intra_weights = 'linear'            # increase intra-domain consistency weights
        inc_warmup = 1

        if not intra_consist:
            stu_tea = False
            stu_stu = False
            proj_lr_const = False

    # ============================================================
    #    Predefined named configurations
    # ============================================================
    @ex.named_config
    def no_self_training():
        freeze_bn = True
        droprate = 0
        use_pseudo_label = False
        warmup_path = None
        sce = False
        cross_consist = False
        intra_consist = False

    @ex.named_config
    def source_only():
        stage = 'source_only'
        lr = 0.0004
        bs = 4
        epochs = 125
    
    @ex.named_config
    def target_only():
        stage = 'target_only'
        lr = 0.0004
        bs = 4
        epochs = 125

    @ex.named_config
    def warmup():
        stage = "warm_up"
        lr = 0.00025
        bs = 4
        epochs = 125
        nesterov = True
        weight_decay = 1e-4

    @ex.named_config
    def synthia():
        name = 'syn2city'
        src_dataset = "synthia"
        src_rootpath = str(DATA_DIR / "SYNTHIA")
        n_class = 16

    @ex.named_config
    def small():
        os = 16
        bs = 4
        lr = 0.0004
        epochs = 125

    @ex.named_config
    def debug():
        epochs = 6
        # stop_stage = 1
        train_n = 8
        test_n = 4
        bs = 2
        print_interval = 2
        val_interval = 4

    # ============================================================
    #    Configurations about observing experiments
    # ============================================================
    @ex.config_hook
    def config_hook(config, command_name, logger):
        add_observers(ex, config, fileStorage=config['fileStorage'], MongoDB=config['mongodb'], db_name=ex.path)
        ex.logger = get_global_logger(name=ex.path)
        return config

    # add missed source files
    ex.add_source_file("data/__init__.py")


def add_observers(ex, config, fileStorage=True, MongoDB=True, db_name="default"):
    if fileStorage:
        observer_file = FileStorageObserver(config["log_dir"])
        ex.observers.append(observer_file)

    if MongoDB:
        try:
            host, port = config["mongo_host"], config["mongo_port"]
            observer_mongo = MongoObserver(url=f"{host}:{port}", db_name=db_name)
            ex.observers.append(observer_mongo)
        except ModuleNotFoundError:
            # Ignore Mongo Observer
            pass


def init_environment(ex, _run, _config, *args, eval_after_train=False):
    configs = [_config] + list(args)
    for i in range(len(configs)):
        configs[i] = MapConfig(configs[i])
    opt = configs[0]
    logger = get_global_logger(name=ex.path)

    if not eval_after_train:
        ### Create experiment directory ###    
        rundir = Path(opt.log_dir) / str(_run._id)
        rundir.mkdir(parents=True, exist_ok=True)
        logger.info(f'RUNDIR: {rundir}')
        _run.rundir = rundir

        ### Backup source code ###
        recover_backup_names(_run)

        ### Reproducbility ###
        set_seed(opt.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    logger.info('Run:' + ' '.join(sys.argv))
    return (*configs, logger, device)


def recover_backup_names(_run):
    if _run.observers:
        for obs in _run.observers:
            if isinstance(obs, FileStorageObserver):
                for source_file, _ in _run.experiment_info['sources']:
                    Path(f'{obs.dir}/source/{source_file}').parent.mkdir(parents=True, exist_ok=True)
                    obs.save_file(source_file, f'source/{source_file}')
                shutil.rmtree(f'{obs.basedir}/_sources')

                # Convert directory `source` to a tarfile `source.tar.gz` for saving server nodes
                with tarfile.open(f"{obs.dir}/source.tar.gz", "w:gz") as t:
                    for root, dir, files in os.walk(f"{obs.dir}/source"):
                        # print(root, dir, files)
                        for f in files:
                            fullpath = os.path.join(root, f)
                            t.add(fullpath, arcname='/'.join(fullpath.split('/')[2:]))
                shutil.rmtree(f'{obs.dir}/source')
                break


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot`
    operation. For example, you can access `cfg["split"]` by `cfg.split`, which
    makes the code more clear. Notice that the result object is a
    sacred.config.custom_containers.ReadOnlyDict, which is a read-only dict for
    preserving the configuration.

    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)
