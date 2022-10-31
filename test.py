import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sacred import Experiment
from pathlib import Path
from PIL import Image

from configs import settings
from data import create_dataset
from metrics import runningScore
from models.puco import PuCo
from utils_ import init_environment, MapConfig, interpb, interpn, get_palette


ex = Experiment("UDASS", base_dir=Path(__file__).parent, save_git_info=False)
settings(ex)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "datasets"

palette = {
    19: get_palette(n_class=19)[0],
    16: get_palette(n_class=16)[0]
}


@ex.config
def config():
    save_pred = False
    save_pred_dir = "predict"

    # PuCo configurations
    cross_consist = False
    intra_consist = False
    if not intra_consist:
        stu_tea = False
        stu_stu = False
    strict = False


def validate(opt, valid_loader, device, model, running_metrics_val, save_dir=None):
    
    for batch in tqdm(valid_loader):
        images_val = batch['image'].to(device)
        labels_val = batch['label'].to(device)
        upsize = labels_val.size()[-2:]

        if opt.save_pred:
            save_paths = [save_dir / (path[:-4] + ".png") for path in batch['image_path']]

        outs = model.BaseNet_DP(images_val)['out']
        outputs = interpb(outs, upsize)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

        if opt.save_pred:
            assert pred.shape[1:] == (1024, 2048), f"{pred.shape}"
            for path, p in zip(save_paths, pred.astype(np.uint8)):
                img = Image.fromarray(p)
                img.putpalette(palette[opt.n_class])
                img.save(path)


# This must be the last function in the script
@ex.automain
def test(_run, _config):
    opt, logger, device = init_environment(ex, _run, _config)

    # create data loader
    datasets = create_dataset(opt, logger)

    # create dataset
    model = PuCo(opt, logger, isTrain=False)
    model.eval(logger=logger)

    # create metric recorder
    running_metrics_val = runningScore(opt.n_class)

    save_dir = None
    if opt.save_pred:
        save_dir = Path(opt.save_pred_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # start evaluation
    with torch.no_grad():
        validate(opt, datasets.target_valid_loader, device, model, running_metrics_val, save_dir)

    # print results        
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        logger.info(f'{k}: {v:.4f}')

    for k, v in class_iou.items():
        logger.info(f'class{k}: {v:.4f}')
