import os
import math
import shutil
import tarfile
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

from utils_ import interpb, get_palette, Timer
from configs import PRE_DIR, PL_DIR


class PseudoLabelSaver(object):
    def __init__(self, opt, logger, device, save_dir, thresh, mode='torch'):
        self.opt = opt
        self.logger = logger
        self.palette = get_palette(opt.n_class)[0]
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        if mode == 'torch':
            self.thresh = torch.from_numpy(thresh[:, None, None]).to(device)
            if torch.any(self.thresh == 1):
                # thresh = 1 will cause missing of the class, because the prob always less than 1. 
                torch.clamp_(self.thresh, 0, 0.999512)
        else:
            self.thresh = thresh[:, None, None]
            if np.any(self.thresh == 1):
                self.thresh = np.clip(self.thresh, 0, 0.999512)

    def save(self, prob, size, save_name):
        """ Generate CBST pseudo labels
        Reference: https://github.com/yzou2/CRST/blob/master/crst_seg.py
        """        
        # Use the pseudo label generating algorithm of CBST
        if self.mode == 'torch':
            weighted_pred = self.save_torch(prob, size)
        else:
            weighted_pred = self.save_np(prob, size)
        # save pseudo-label map with label IDs
        pseudo_label_save = Image.fromarray(weighted_pred)
        pseudo_label_save.putpalette(self.palette)
        save_path = self.save_dir / save_name
        pseudo_label_save.save(save_path)

    def save_np(self, prob, size):
        prob = ndi.zoom(prob, (1, size[0] / prob.shape[-2], size[1] / prob.shape[-1]), order=1)
        weighted_prob = prob / self.thresh
        weighted_conf = np.max(weighted_prob, axis=0)
        weighted_pred = np.asarray(np.argmax(weighted_prob, axis=0), np.uint8)
        weighted_pred[weighted_conf < 1] = 250      # 250 is ignored in loss functions
        return weighted_pred

    def save_torch(self, prob, size):
        prob = interpb(prob.unsqueeze(0), size).squeeze(0)
        weighted_prob = prob / self.thresh
        weighted_conf, weighted_pred = torch.max(weighted_prob, dim=0)
        weighted_pred[weighted_conf < 1] = 250      # 250 is ignored in loss functions
        return weighted_pred.byte().cpu().numpy()

    def make_tar(self, moveto=None):
        self.logger.info(f"Making tarfile: {self.save_dir}.tar")
        with tarfile.open(f"{self.save_dir}.tar", "w") as t:
            for root, dir, files in os.walk(f"{self.save_dir}"):
                for f in files:
                    fullpath = os.path.join(root, f)
                    t.add(fullpath, arcname='/'.join(fullpath.split('/')[2:]))
        if moveto:
            try:
                shutil.move(f"{self.save_dir}.tar", str(moveto))
                self.logger.info(f"Move {self.save_dir}.tar to {moveto}")
            except Exception as e:
                print(e)


def inference_all(model, datasets, device, soft=True, hard=True):
    softs, confs, preds, image_paths = [], [], [], []   # we save all the probs, although it consumes a lot of memory 
    with torch.no_grad():
        for batch in datasets.target_pl_loader:
            images_val = batch['image'].to(device)
            out = F.softmax(model.BaseNet_DP(images_val)['out'], dim=1)

            for img_p in batch['image_path']:
                image_paths.append(img_p)
            # ------------------------------------------------------------------------
            # save soft prediction
            if soft:
                for prob in out.cpu().numpy(): 
                    softs.append(prob)

            # ------------------------------------------------------------------------
            # save hard prediction
            if hard:
                upsize = images_val.size()[2:]
                out = interpb(out, upsize)
                flip_out = F.softmax(model.BaseNet_DP(images_val.flip([-1]))['out'], dim=1)
                flip_out = interpb(flip_out, upsize).flip([-1])            
                out = (out + flip_out) / 2

                confidence, pseudo = out.max(1)
                for conf in confidence.half().cpu().numpy():
                    confs.append(conf)      # float16
                for pred in pseudo.byte().cpu().numpy():
                    preds.append(pred)      # uint8
    return softs, confs, preds, image_paths


def generate_kc_parameters(opt, logger, conf_dict, pred_cls_num):
    logger.info("###### Start kc generation! ######")
    logger.info("Please be patient. Sorting pixels...")
    timer = Timer()

    # Compute multiple pseudo label percentages for convenience
    tgt_ports = np.linspace(0, 1, 21)[1:]
    cls_thresh = {f'{k:.2f}': np.ones(opt.n_class, dtype = np.float32) for k in tgt_ports}

    with timer.start():
        for c in np.arange(opt.n_class):
            logger.info(f"Processing class{c}")
            if conf_dict[c]:
                # sort in descending order, the most time-costing part
                conf_dict[c].sort(reverse=True)
                len_cls = len(conf_dict[c])

                # get the threshold for each percentage
                for k in tgt_ports:
                    len_cls_thresh = int(math.floor(len_cls * k))
                    if len_cls_thresh != 0:
                        cls_thresh[f'{k:.2f}'][c] = conf_dict[c][len_cls_thresh - 1]
                del conf_dict[c]

    for k, v in cls_thresh.items():
        print(k, '[' + ' '.join([f'{x:.6f}' for x in v]) + ']')

    return cls_thresh
    

def generate_pseudo_label(opt, logger, device, model, datasets, thresh, status_path, pl_dir, pl_tar_path):
    """ Pseudo Label Generation
        
        Step 1: Compute predictions
        Step 2: Compute CBST thresholds from predictions
        Step 3: Generate class-balanced pseudo labels with CBST thresholds

        Make sure you have at least 60G available system memory
    """
    ###  Step 1: Compute predictions
    logger.info("Step 1: Compute predictions")
    upsize = datasets.target_pl.img_size
    model.eval(logger=logger)
    softs, confs, preds, image_paths = inference_all(model, datasets, device)

    ###  Step 2: Compute CBST thresholds from predictions
    logger.info("Step 2: Compute CBST thresholds from predictions")
    timer = Timer()
    conf_dict = {k: [] for k in range(opt.n_class)}
    pred_cls_num = np.zeros(opt.n_class, np.int64)      # too many elements, int32 will overflow
    with timer.start():
        for conf, pred in zip(confs, preds):
            for c in range(opt.n_class):
                idx_temp = pred == c
                pred_cls_num[c] = pred_cls_num[c] + np.sum(idx_temp)
                if idx_temp.any():
                    conf_cls_temp = conf[idx_temp].astype(np.float32)
                    len_cls_temp = conf_cls_temp.size
                    # downsampling by ds_rate
                    conf_cls = conf_cls_temp[0:len_cls_temp:opt.ds_rate]
                    conf_dict[c].extend(conf_cls)
    logger.info(f"Finish extracting confidence of target domain train set! Time cost: {timer.diff:.2f} seconds.")
    logger.info(f"Class pixels of predictions: {pred_cls_num}")
    all_cls_thresh = generate_kc_parameters(opt, logger, conf_dict, pred_cls_num)
    try:
        np.save(status_path, all_cls_thresh)
        logger.info(f'Finish kc generation! Time cost: {timer.diff:.2f} seconds.')
    except Exception as e:
        print(e)

    ### Step 3: Generate class-balanced pseudo labels with CBST thresholds
    logger.info("Step 3: Generate class-balanced pseudo labels with CBST thresholds")
    cls_thresh = all_cls_thresh[f'{thresh:.2f}']
    saver = PseudoLabelSaver(opt, logger, device, pl_dir, cls_thresh, 'np')
    for prob_i, img_p in zip(softs, image_paths):
        saver.save(prob_i, upsize, Path(img_p).stem + '.png')
    # make tar file for future usage
    saver.make_tar(moveto=pl_tar_path)


def maybe_generate_pseudo_label(opt, logger, device, model, datasets, stage, expID):
    ### Skip generation if exists ###
    thresh = opt.thresh[stage - 1]
    pl_name = f"{opt.task_name}_s{stage}_{thresh:.2f}"
    local_pl_dir = PL_DIR / pl_name
    pl_tar_path = PRE_DIR / (pl_name + ".tar")
    if stage > 1: 
        local_pl_dir = PL_DIR / expID / pl_name
        pl_tar_path = PRE_DIR / expID / (pl_name + ".tar")

    # maybe extract from tar file
    if not local_pl_dir.exists() and pl_tar_path.exists():
        local_pl_dir.parent.mkdir(parents=True, exist_ok=True)
        os.system(f'tar -xf {pl_tar_path} -C {local_pl_dir.parent}/')

    if local_pl_dir.exists():
        logger.info("#========================================================")
        logger.info(f"Reusing pseudo labels")
        logger.info("#========================================================")
        return local_pl_dir

    ### Load best weights
    if stage > 1:
        best_model_path = PRE_DIR / expID / "best_model.pth"
        model.load_base_weights(model.BaseNet, best_model_path, init=False)

    ### Skip computing thresholds if exists ###
    status_file = f"{opt.task_name}_s{stage}_cls_thresh.npy"
    status_path = PRE_DIR / status_file
    if stage > 1: 
        status_path = PRE_DIR / expID / status_file
    if status_path.exists():
        logger.info("#========================================================")
        logger.info(f"Generating pseudo labels from status")
        logger.info("#========================================================")
        cls_thresh = np.load(status_path, allow_pickle=True).item()[f'{thresh:.2f}']
 
        # generating pseudo labels with cls_thresh
        model.eval(logger=logger)
        upsize = datasets.target_pl.img_size
        with torch.no_grad():
            saver = PseudoLabelSaver(opt, logger, device, local_pl_dir, cls_thresh, 'torch')
            for batch in datasets.target_pl_loader:
                images_val = batch['image'].to(device)
                out = F.softmax(model.BaseNet_DP(images_val)['out'], dim=1)
                for prob_i, img_p in zip(out, batch['image_path']):
                    saver.save(prob_i, upsize, Path(img_p).stem + '.png')
        # make tar file for future usage
        saver.make_tar(moveto=pl_tar_path)
        
        ### Restore last weights
        if stage > 1:
            if opt.use_best:
                model.load_ema_weights(model.BaseNet_ema, model.BaseNet)
            else:
                last_model_path = PRE_DIR / expID / "model.pth"
                model.load_base_weights(model.BaseNet, last_model_path, init=False)

        return local_pl_dir

    ### Generate pseudo labels from start
    logger.info("#========================================================")
    logger.info(f"Generating pseudo labels from start")
    generate_pseudo_label(opt, logger, device, model, datasets, thresh, status_path, local_pl_dir, pl_tar_path)
    logger.info("#========================================================")

    ### Restore last weights
    if opt.use_best:
        model.load_ema_weights(model.BaseNet_ema, model.BaseNet)
    else:
        last_model_path = PRE_DIR / expID / "model.pth"
        model.load_base_weights(model.BaseNet, last_model_path, init=False)

    return local_pl_dir
