import random
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment

from loggers import C as CC
from data import create_dataset
from metrics import runningScore
from models.puco import PuCo
from configs import init_environment, settings, PRE_DIR
from constants import on_cloud
from utils_ import Accumulator, Timer, interpb, save_ckpt, maybe_extract_datasets
from utils_pl import maybe_generate_pseudo_label

ex = Experiment("PuCo", base_dir=Path(__file__).parent, save_git_info=False)
settings(ex)


def validation(model, logger, datasets, device, running_metrics_val, iters, epoch, opt=None, run=None):
    for _k, v in enumerate(model.optimizers):
        for kk, param_group in enumerate(v.param_groups):
            _learning_rate = param_group.get('lr')
            logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))
            run.log_scalar(f'lr{kk + 1}', _learning_rate, iters)

    # ==============================================================================
    ### Evaluation ###
    model.eval(logger=logger)
    loader = datasets.target_valid_loader
    with torch.no_grad():
        for batch in loader:
            images_val = batch['image'].to(device)
            labels_val = batch['label'].to(device)
            
            out = model.BaseNet_DP(images_val)
            outputs = interpb(out['out'], size=labels_val.size()[-2:])

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics_val.update(gt, pred)

    # ==============================================================================
    ### Save metrics ###
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        logger.info(f'{k}: {v}')
        run.log_scalar(k, float(v), iters)

    for k, v in class_iou.items():
        logger.info(f'{k:2d} {loader.dataset.cname(k):15s}: {v:.4f}')
        run.log_scalar(f"class{k}", float(v), iters)

    is_best = False
    if score["Mean IoU : \t"] >= model.best_iou:
        model.best_iou = score["Mean IoU : \t"]
        is_best = True

    # ==============================================================================
    ### Collect model states ###
    state = {
        'iter': iters,
        'epoch': epoch,
        'iou': score["Mean IoU : \t"],
        'best_iou': model.best_iou
    }
    for _k, net in enumerate(model.nets):
        new_state = {"model_state": net.state_dict()}
        state[net.name_] = new_state

    curr_metric = score["Mean IoU : \t"]
    running_metrics_val.reset()
    rundir = Path(run.rundir)

    ### Save checkpoints ###
    save_path = rundir / "model.pth"
    save_ckpt(state, save_path)
    if opt.stu_tea:
        state_ema = {model.BaseNet_ema.name_: {"model_state": model.BaseNet_ema.state_dict()}}
        save_path_ema = rundir / "model_ema.pth"
        save_ckpt(state_ema, save_path_ema)
    if is_best:
        save_ckpt(save_path, rundir / "best_model.pth")
        if opt.stu_tea:
            save_ckpt(save_path_ema, rundir / "best_model_ema.pth")

    # ==============================================================================
    if opt.save_all and rundir.name != 'None':
        cloud_save_dir = PRE_DIR / rundir.name
        if not cloud_save_dir.exists():
            cloud_save_dir.mkdir(parents=True)

        # Collect all states
        for _k, net in enumerate(model.nets):
            state[net.name_].update({
                "optimizer_state": model.optimizers[_k].state_dict(),
                "scheduler_state": model.schedulers[_k].state_dict(),
            })
        state['random'] = random.getstate()
        state['np.random'] = np.random.get_state()
        state['torch.random'] = torch.get_rng_state().numpy()

        # Save states on cloud
        try:
            save_ckpt(state, cloud_save_dir / save_path.name, cloud=True)
            if opt.stu_tea:
                save_ckpt(save_path_ema, cloud_save_dir / save_path_ema.name, cloud=True)
            if is_best:
                save_ckpt(save_path, cloud_save_dir / "best_model.pth", cloud=True)
                if opt.stu_tea:
                    save_ckpt(save_path_ema, cloud_save_dir / "best_model_ema.pth", cloud=True)
        except Exception as e:
            print(e)

    return curr_metric


# This must be the last function in the script
@ex.automain
def train(_run, _config):
    ### Initialize environment ###
    opt, logger, device = init_environment(ex, _run, _config)

    ### Construct model ###
    # Try to restore from pretrained directory
    resume_path = PRE_DIR / _run.rundir.name / "model.pth"
    if not resume_path.exists():
        resume_path = None
    model = PuCo(opt, logger, resume_path)

    ### Construct datasets ###
    if on_cloud: maybe_extract_datasets(opt, logger)
    datasets = create_dataset(opt, logger)

    ### Construct optimizer and scheduler ###
    model.stage_epochs = opt.epochs // opt.nstages
    model.train_iters = opt.epochs * len(datasets.target_train_loader)
    model.stage_iters = model.stage_epochs * len(datasets.target_train_loader)
    model.optimizer_and_scheduler()

    ### Setup Metrics ###
    timer = Timer()
    running_metrics_val = runningScore(opt.n_class)
    accu = Accumulator(**{x: 0. for x in model.loss_names})

    for epoch in range(model.epoch, opt.epochs + 1):
        # Early stop by stages
        if opt.stop_stage and epoch > opt.stop_stage * model.stage_epochs:
            break
        # Start a new stage, generate pseudo-labels
        if opt.use_pseudo_label and (epoch == model.epoch or epoch % model.stage_epochs == 1):
            stage = (epoch - 1) // model.stage_epochs + 1
            pseudo_label_dir = maybe_generate_pseudo_label(opt, logger, device, model, datasets, stage, _run.rundir.name)
            # target train loader use new pseudo labels
            datasets.target_train.pseudo_label_base = pseudo_label_dir
            logger.info(CC.c(f'Load pseudo labels from {pseudo_label_dir}', CC.OKBLUE))
            
            # Reset lr for new stages
            if stage > 1 and epoch % model.stage_epochs == 1:
                model.reset_lr_and_scheduler()
                logger.info("Reset training states for next stage.")

        model.epoch = epoch
        for target_batch in datasets.target_train_loader:
            model.iter += 1
            i = model.iter

            ### Target inputs ###
            target_image = target_batch['image'].to(device)
            target_label = target_batch['label'].to(device)
            target_image2 = target_batch['image2'].to(device) if 'image2' in target_batch else None
            target_pseudo_label = target_batch['pseudo_label'].to(device) if 'pseudo_label' in target_batch else None
            kwargs1 = target_batch['kwargs1'] if 'kwargs1' in target_batch else None
            kwargs2 = target_batch['kwargs2'] if 'kwargs2' in target_batch else None

            ### Source inputs ###
            source_batch = datasets.source_train_loader.next()
            images = source_batch['image'].to(device)
            labels = source_batch['label'].to(device)
            
            ### 1. Train step ###
            with timer.start():
                model.train()
                losses = model(images, labels, target_image, target_image2, target_label, target_pseudo_label, 
                               kwargs1, kwargs2, epoch)
                accu.update(**losses)

            ### Print losses ###
            model.maybe_print_losses(accu, timer)

            ### 2. Validation step ###
            if opt.val_interval > 0 and i % opt.val_interval == 0:
                # Log to Sacred
                for k, v in accu.mean(model.loss_names, dic=True).items():
                    _run.log_scalar(k, v, model.iter)
                accu.reset()

                validation(model, logger, datasets, device, running_metrics_val, iters=model.iter, epoch=epoch, opt=opt, run=_run)
                logger.info('Best iou until now is {}'.format(model.best_iou))

            model.scheduler_step()

    return f"Mean IoU: {model.best_iou:.4f}"
