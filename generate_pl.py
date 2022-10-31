import math
from itertools import cycle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from sacred import Experiment
from tqdm import tqdm

from configs import settings, PL_DIR
from data import create_dataset
from models.puco import PuCo
from utils_ import init_environment, Timer, MapConfig, interpb, get_palette

ex = Experiment("UDASS", base_dir=Path(__file__).parent, save_git_info=False)
settings(ex)

palette = {
    19: get_palette(n_class=19)[0],
    16: get_palette(n_class=16)[0]
}


@ex.config
def configuration():
    task_name = "gta2city"          # pretrain source model
    soft = False                    # save soft pseudo label
    droplast = False                # overlap the default setting
    shuffle = False                 # overlap the default setting
    aug = False                     # overlap the default setting
    tag = "warmup"                  # tag for inditifying pseudo label versions
    pl_dir = str(PL_DIR)            # directory to save all the pseudo labels

    # PuCo configurations
    cross_consist = False
    intra_consist = False
    if not intra_consist:
        stu_tea = False
        stu_stu = False

    # CBST configurations
    use_pseudo_label = False        # disable loading pseudo labels in this script
    ds_rate = 4                     # The downsampling rate in kc calculation.
    thresh = 0.2                    # CBST thresh
    soft_dir = f"{pl_dir}/{task_name}_{tag}_soft"                   # path to soft prediction
    hard_dir = f"{pl_dir}/{task_name}_{tag}_hard"                   # path to hard prediction
    status_dir = f"{pl_dir}/{task_name}_{tag}_status"               # path to class thresh status
    pseudo_label_dir = f"{pl_dir}/{task_name}_{tag}_{thresh:.2f}"   # path to pseudo labels
    

@ex.main
def generate_pseudo_label(_run, _config):
    """ Pseudo Label Generation
        
        Step 1: Save predictions
        Step 2: Compute CBST thresholds from predictions
        Step 3: Generate class-balanced pseudo labels with CBST thresholds

    """
    opt, logger, device = init_environment(ex, _run, _config)

    print("""
    # =========================================================== #
    #                                                             #
    #   Make sure you have at least 60G available system memory   #
    #                                                             #
    # =========================================================== #
    """)

    # create save directory
    soft_dir = Path(opt.soft_dir)
    hard_dir = Path(opt.hard_dir)
    status_dir = Path(opt.status_dir)
    pseudo_label_dir = Path(opt.pseudo_label_dir)
    logger.info(f"Soft directory:         {soft_dir}")
    logger.info(f"Hard directory:         {hard_dir}")
    logger.info(f"Status directory:       {status_dir}")
    logger.info(f"Pseudo label directory: {pseudo_label_dir}")
    logger.info(f"Load checkpoint from:   {opt.resume_path or opt.warmup_path}")
    v = input("Confirm: [Y]/n ")
    if v not in ["", "Y", "y", "Yes", "yes", "YES"]:
        return
    soft_dir.mkdir(exist_ok=True, parents=True)
    hard_dir.mkdir(exist_ok=True, parents=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    pseudo_label_dir.mkdir(parents=True, exist_ok=True)

    # =======================================================================================
    #     Step 1: Save predictions
    # =======================================================================================
    datasets = create_dataset(opt, logger)
    model = PuCo(opt, logger, isTrain=False)
    model.eval(logger=logger)

    with torch.no_grad():
        validate(opt, logger, model, datasets.target_train_loader, device, soft_dir, hard_dir)

    # Release memory
    del model
    del datasets
    torch.cuda.empty_cache() 

    # =======================================================================================
    #     Step 2: Compute CBST thresholds from predictions
    # =======================================================================================
    timer = Timer()

    # saving output data
    conf_dict = {k: [] for k in range(opt.n_class)}
    pred_cls_num = np.zeros(opt.n_class, np.int64)      # too many elements, int32 will overflow

    # collect prediction confidence files
    all_conf_files = list(hard_dir.glob("*_conf.npy"))

    with timer.start():
        for conf_file in tqdm(all_conf_files):
            conf = np.load(conf_file)   # [1024, 2048]
            pred_file = hard_dir / conf_file.name.replace("_conf.npy", ".png")
            pred = np.array(Image.open(pred_file), np.uint8)

            for c in range(opt.n_class):
                idx_temp = pred == c
                pred_cls_num[c] = pred_cls_num[c] + np.sum(idx_temp)
                if idx_temp.any():
                    conf_cls_temp = conf[idx_temp].astype(np.float32)
                    len_cls_temp = conf_cls_temp.size
                    # downsampling by ds_rate
                    conf_cls = conf_cls_temp[0:len_cls_temp:opt.ds_rate]
                    conf_dict[c].extend(conf_cls)

    logger.info(f'###### Finish extracting confidence of target domain train set! '
                f'Time cost: {timer.diff:.2f} seconds. ######')

    print("Class pixels of predictions:", pred_cls_num)
    all_cls_thresh = generate_kc_parameters(conf_dict, pred_cls_num, status_dir, opt, logger)

    # =======================================================================================
    #     Step 3: Generate class-balanced pseudo labels with CBST thresholds
    # =======================================================================================
    cls_thresh = all_cls_thresh[f'{opt.thresh:.2f}']

    p = Pool(8)

    all_files = list(sorted(soft_dir.glob("*.npy")))
    list(tqdm(p.imap(save_by_confidence, 
                    zip(all_files,
                        cycle([hard_dir]),
                        cycle([pseudo_label_dir]),
                        cycle([cls_thresh[:, None, None]]))), total=len(all_files)))


def val_save(args):
    save_path, pseudo, confidence, n_class = args
    
    # save hard prediction
    pil_img = Image.fromarray(pseudo)
    pil_img.putpalette(palette[n_class])
    pil_img.save(save_path)

    # save prediction confidence
    np.save(save_path.parent / save_path.name.replace('.png', '_conf.npy'), confidence)


def val_save_soft(args):
    save_path, prob = args
    
    # save soft prediction
    np.save(save_path.with_suffix('.npy'), prob)


def validate(opt, logger, model, valid_loader, device, soft_dir, hard_dir):
    p = Pool(8)
    sm = torch.nn.Softmax(dim=1)

    for batch in tqdm(valid_loader):
        images_val = batch['image'].to(device)
        labels_val = batch['label'].to(device)
        upsize = images_val.size()[2:]

        out = sm(model.BaseNet_DP(images_val)['out'])

        # ------------------------------------------------------------------------
        # save soft prediction
        prob = out.cpu().numpy()
        soft_names = [soft_dir / Path(path).name for path in batch['image_path']]
        p.map(val_save_soft, zip(soft_names, prob))

        # ------------------------------------------------------------------------
        # save hard prediction
        out = interpb(out, upsize)

        flip_out = sm(model.BaseNet_DP(images_val.flip([-1]))['out'])
        flip_out = interpb(flip_out, upsize).flip([-1])
        
        out = (out + flip_out) / 2

        confidence, pseudo = out.max(1)
        pseudo = pseudo.cpu().numpy().astype(np.uint8)
        confidence = confidence.cpu().numpy().astype(np.float16)

        hard_names = [hard_dir / Path(path).name for path in batch['image_path']]
        p.map(val_save, zip(hard_names, pseudo, confidence, cycle([opt.n_class])))


def generate_kc_parameters(conf_dict, pred_cls_num, save_stats_dir, opt, logger):
    logger.info("###### Start kc generation! ######")
    logger.info("Please be patient. Sorting pixels...")
    timer = Timer()

    # Compute multiple pseudo label percentages for convenience
    tgt_ports = np.linspace(0, 1, 21)[1:]
    cls_thresh = {f'{k:.2f}': np.ones(opt.n_class, dtype = np.float32) for k in tgt_ports}

    with timer.start():
        for c in tqdm(np.arange(opt.n_class)):
            if conf_dict[c]:
                # sort in descending order, the most time-costing part
                conf_dict[c].sort(reverse=True)
                len_cls = len(conf_dict[c])

                # get the threshold for each percentage
                for k in tgt_ports:
                    len_cls_thresh = int(math.floor(len_cls * k))
                    if len_cls_thresh != 0:
                        cls_thresh[f'{k:.2f}'][c] = conf_dict[c][len_cls_thresh - 1]
                conf_dict[c] = None

    np.save(save_stats_dir / 'cls_thresh.npy', cls_thresh)
    logger.info(f'###### Finish kc generation! Time cost: {timer.diff:.2f} seconds. ######')

    for k, v in cls_thresh.items():
        print(k, '[' + ' '.join([f'{x:.6f}' for x in v]) + ']')

    return cls_thresh


def save_by_confidence(args):
    """ Generate CBST pseudo labels

    Reference: https://github.com/yzou2/CRST/blob/master/crst_seg.py
    """
    soft_file, hard_dir, save_dir, thresh_ = args
    
    prob = np.load(soft_file)
    w, h = Image.open(hard_dir / soft_file.with_suffix(".png").name).size
    prob = ndi.zoom(prob, (1, h / prob.shape[-2], w / prob.shape[-1]), order=1)
    n_class = prob.shape[0]
    
    # Use the pseudo label generating algorithm of CBST
    if np.any(thresh_ == 1):
        # thresh = 1 will cause missing of the class, because the prob always less than 1. 
        thresh_ = np.clip(thresh_, 0, 0.999512)
    weighted_prob = prob / thresh_
    weighted_conf = np.max(weighted_prob, axis=0)
    weighted_pred = np.asarray(np.argmax(weighted_prob, axis=0), np.uint8)
    weighted_pred[weighted_conf < 1] = 250      # 250 is ignored in loss functions

    # save pseudo-label map with label IDs
    pseudo_label_save = Image.fromarray(weighted_pred)
    pseudo_label_save.putpalette(palette[n_class])
    save_path = save_dir / soft_file.with_suffix(".png").name
    pseudo_label_save.save(save_path)


# ====================================================================================


@ex.command
def generate_pseudo_label_step3(_run, _config):
    """
    This command is used to generate pseudo labels with specified CBST percentages.
    It only contains the `Step 3`.

    """
    opt, logger, device = init_environment(ex, _run, _config)

    cls_thresh_file = Path(opt.status_dir) / "cls_thresh.npy"
    all_cls_thresh = np.load(cls_thresh_file, allow_pickle=True).item()
    cls_thresh = all_cls_thresh[f'{opt.thresh:.2f}']

    # create save directory
    soft_dir = Path(opt.soft_dir)
    hard_dir = Path(opt.hard_dir)
    status_dir = Path(opt.status_dir)
    pseudo_label_dir = Path(opt.pseudo_label_dir)
    logger.info(f"Soft directory:         {soft_dir}")
    logger.info(f"Hard directory:         {hard_dir}")
    logger.info(f"Status directory:       {status_dir}")
    logger.info(f"Pseudo label directory: {pseudo_label_dir}")
    input("Confirm:")
    pseudo_label_dir.mkdir(parents=True, exist_ok=True)

    p = Pool(8)

    all_files = list(sorted(soft_dir.glob("*.npy")))
    list(tqdm(p.imap(save_by_confidence, 
                    zip(all_files,
                        cycle([hard_dir]),
                        cycle([pseudo_label_dir]),
                        cycle([cls_thresh[:, None, None]]))), total=len(all_files)))


@ex.command(unobserved=True)
def show_status(_config):
    opt = MapConfig(_config)

    cls_thresh_file = Path(opt.status_dir) / "cls_thresh.npy"
    cls_thresh = np.load(cls_thresh_file, allow_pickle=True).item()
    for k, v in cls_thresh.items():
        print(k, '[' + ' '.join([f'{x:.6f}' for x in v]) + ']')


if __name__ == "__main__":
    ex.run_commandline()
