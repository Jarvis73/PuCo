import random
import numpy as np
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path

from models.deeplabv2 import Deeplab
from models.discriminator import FCDiscriminator
from utils_ import (
    get_scheduler, cross_entropy2d, interpb, interpn
)
from loggers import C as CC


def _freeze_bn(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


class PuCo(object):
    def __init__(self, opt, logger, resume_path=None, isTrain=True):
        self.opt = opt
        self.n_class = opt.n_class
        self.logger = logger
        self.isTrain = isTrain
        self.best_iou = 0
        self.nets = []
        self.nets_DP = []
        self.default_gpu = 0
        self.num_devices = torch.cuda.device_count()
        self.device = torch.device(
            "cuda:{}".format(self.default_gpu) if torch.cuda.is_available() else 'cpu')

        ### Define segmentation Base(student) model ###
        self.BaseNet = Deeplab(nn.BatchNorm2d, self.opt.os, mtype='Base', opt=opt)
        resume_path = resume_path or opt.resume_path
        if resume_path is not None:
            resume_path = Path(resume_path)
        self.checkpoint = self.load_base_weights(self.BaseNet, resume_path)
        self.nets.extend([self.BaseNet])
        self.BaseNet_DP = self.init_device(self.BaseNet)
        self.nets_DP.append(self.BaseNet_DP)
        logger.info('the backbone is {}'.format(opt.model_name))

        ### Define EMA(teacher) model ###
        if opt.stu_tea:
            self.BaseNet_ema = Deeplab(
                nn.BatchNorm2d, self.opt.os, mtype='EMA', opt=opt)
            resume_path_ema = resume_path.with_name(resume_path.stem + '_ema.pth') if resume_path is not None else None
            self.load_ema_weights(self.BaseNet_ema, self.BaseNet, resume_path_ema)
            self.BaseNet_ema_DP = self.init_device(self.BaseNet_ema)
            self.BaseNet_ema_DP.eval()

        self.adv_source_label = 0
        self.adv_target_label = 1
        self.bceloss = torch.nn.MSELoss()

        self.loss_names = self._get_loss_names(opt)
        self.iter, self.epoch = self.restore_random_states()
        self.train_iters = 0    # set in caller
        self.stage_epochs = 0   # set in caller
        self.stage_iters = 0    # set in caller

        # Enable AMP
        if opt.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def load_base_weights(self, model, resume_path, mtype='Base', init=True):
        opt = self.opt
        ### Initialize model parameters ###
        if init:
            self.logger.info(CC.c(f"[{mtype}] Model parameters initialized from {opt.init_path}", CC.BOLD))
            pretrain_dict = torch.load(opt.init_path, map_location='cpu')
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in pretrain_dict.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)

        ### Restore model parameters from warmup model or checkpoints ###
        if resume_path is not None:
            ### Continue training ###
            self.logger.info(CC.c(f"[{mtype}] Checkpoint resumed from {resume_path}", CC.BOLD))
            checkpoint = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(checkpoint['ResNet101']["model_state"])
            return checkpoint

        if init and opt.warmup_path is not None:
            ### Restore from the warmup model ###
            self.logger.info(CC.c(f"[{mtype}] Checkpoint initialized from {opt.warmup_path}", CC.BOLD))
            checkpoint = torch.load(opt.warmup_path, map_location='cpu')
            model.load_state_dict(checkpoint['ResNet101']["model_state"], strict=opt.strict)

    def load_ema_weights(self, model_ema, model, resume_path_ema=None):
        ### Restore EMA model parameters ###
        if resume_path_ema is not None:
            # Continue training
            checkpoint = torch.load(resume_path_ema, map_location='cpu')
            try:
                model_ema.load_state_dict(checkpoint['ResNet101']['model_state'])
            except Exception as e:
                self.logger.info(CC.c(f"[EMA] Checkpoint restored from {resume_path_ema}", CC.BOLD))
                print(e)
                raise e
            self.logger.info(CC.c(f"[EMA] Checkpoint restored from {resume_path_ema}", CC.BOLD))
        else:
            # Copy from BaseNet
            model_dict = {}
            state_dict = model_ema.state_dict()
            for k, v in model.state_dict().items():
                if k in state_dict:
                    model_dict[k] = v
            model_ema.load_state_dict(model_dict)
            self.logger.info(CC.c(f"[EMA] Checkpoint copyed from BaseNet", CC.BOLD))

    def _get_loss_names(self, opt):
        if opt.stage == 'source_only':
            loss_lst = ['src']
        elif opt.stage == 'target_only':
            loss_lst = ['tgt']
        elif opt.stage == 'warm_up':
            loss_lst = 'src G D'.split()
        elif opt.stage == 'self_training':
            loss_lst = 'src src_style loss tgt stu_tea stu_stu'.split()            # [1]
        else:
            raise ValueError
        
        return loss_lst

    def optimizer_and_scheduler(self, restore=True):
        opt = self.opt

        ### Define optimizer ###
        self.optimizers = []
        optimizer_params = {
            'lr': opt.lr,
            'weight_decay': opt.weight_decay,
            'momentum': opt.momentum,
            'nesterov': opt.weight_decay
        }
        self.BaseOpti = torch.optim.SGD(
            self.BaseNet.get_param_list(opt.stage, optimizer_params['lr'], opt.proj_lr_const), 
            **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        ### Define scheduler ###
        self.schedulers = []
        assert self.stage_iters > 0, "self.stage_iters must be set before calling self.optimizer_and_scheduler()"
        self.BaseSchedule = get_scheduler(self.BaseOpti, 
                                        self.stage_iters,
                                        gamma=0.9,
                                        proj_lr_const=opt.proj_lr_const)
        self.schedulers.extend([self.BaseSchedule])
        
        ### Restore optimizer/scheduler state ###
        if restore and self.checkpoint is not None:
            self.BaseOpti.load_state_dict(self.checkpoint['ResNet101']['optimizer_state'])
            self.BaseSchedule.load_state_dict(self.checkpoint['ResNet101']['scheduler_state'])

        ### Define discriminator ###
        if opt.stage == "warm_up":
            self.net_D = FCDiscriminator(inplanes=self.n_class)
            self.net_D_DP = self.init_device(self.net_D)
            self.nets.extend([self.net_D])
            self.nets_DP.append(self.net_D_DP)

            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
            self.optimizers.extend([self.optimizer_D])
            
            self.DSchedule = get_scheduler(self.optimizer_D, opt)
            self.schedulers.extend([self.DSchedule])

            ### Restore discriminator parameters ###
            if restore and self.checkpoint is not None:
                self.net_D.load_state_dict(self.checkpoint['FCDiscriminator']['model_state'])
                self.optimizer_D.load_state_dict(self.checkpoint['FCDiscriminator']['optimizer_state'])
                self.DSchedule.load_state_dict(self.checkpoint['FCDiscriminator']['scheduler_state'])

    def reset_lr_and_scheduler(self):
        opt = self.opt
        # Reset optimizer momentum and learning rate
        if opt.reset_all:
            self.optimizer_and_scheduler(restore=False)
            return
        
        # Reset learning rate
        for group in self.BaseOpti.param_groups:
            group['lr'] = opt.lr
        # Reset learning rate scheduler
        self.schedulers[0] = get_scheduler(self.BaseOpti, 
                                        self.stage_iters,
                                        gamma=0.9,
                                        proj_lr_const=opt.proj_lr_const)
        if opt.stage == "warm_up":
            for group in self.optimizer_D.param_groups:
                group['lr'] = 1e-4
            self.schedulers[1] = get_scheduler(self.optimizer_D, opt)

    def restore_random_states(self):
        if self.checkpoint:
            random.setstate(self.checkpoint['random'])
            np.random.set_state(self.checkpoint['np.random'])
            torch.set_rng_state(torch.from_numpy(self.checkpoint['torch.random']).byte())
            iter_ = self.checkpoint['iter']
            epoch = self.checkpoint['epoch'] + 1
            self.best_iou = self.checkpoint['best_iou']
            self.logger.info(CC.c(f"Random states restored. Start from epoch {epoch}, iter {iter_ + 1}", CC.BOLD))
        else:
            iter_ = 0
            epoch = 1

        return iter_, epoch

    def _forward(self, model, *args, **kwargs):
        return model(*args, **kwargs)

    def _backward(self, loss):
        if self.opt.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimize(self):
        if self.opt.amp:
            self.scaler.step(self.BaseOpti)
            self.scaler.update()
        else:
            self.BaseOpti.step()

    def __call__(self, images, labels, target_image, target_image2, target_label, target_pseudo_label, 
                 kwargs1, kwargs2, epoch):
        if self.opt.stage == 'source_only':
            losses = self.step_source_only(images, labels)
        elif self.opt.stage == 'target_only':
            losses = self.step_target_only(target_image, target_label)
        elif self.opt.stage == 'warm_up':
            losses = self.step_warm_up(images, labels, target_image)
        elif self.opt.stage == 'self_training':
            losses = self.step_self_training(images, labels, target_image, target_image2, target_pseudo_label, 
                                             kwargs1, kwargs2, epoch)
        else:
            raise ValueError
        
        return losses

    def step_source_only(self, source_x, source_label):
        source_out = self.BaseNet_DP(source_x)
        source_out['out'] = interpb(source_out['out'], source_x.size()[2:])
        loss_GTA = cross_entropy2d(inputs=source_out['out'], target=source_label)
        loss_GTA.backward()

        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        return {
            "src": loss_GTA.item()
        }

    def step_target_only(self, target_x, target_label):
        target_out = self.BaseNet_DP(target_x)
        target_out['out'] = interpb(target_out['out'], size=target_x.size()[2:])
        loss_CTS = cross_entropy2d(inputs=target_out['out'], target=target_label)
        loss_CTS.backward()

        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        return {
            'tgt': loss_CTS.item()
        }

    def step_warm_up(self, source_x, source_label, target_x):
        for param in self.net_D.parameters():
            param.requires_grad = False
        self.BaseOpti.zero_grad()
        
        source_output = self.BaseNet_DP(source_x)
        source_outputUp = interpb(source_output['out'], source_x.size()[2:])
        loss_GTA = cross_entropy2d(inputs=source_outputUp, target=source_label)

        target_output = self.BaseNet_DP(target_x)
        target_outputUp = interpb(target_output['out'], target_x.size()[2:])
        target_D_out = self.net_D_DP(F.softmax(target_outputUp, dim=1))
        loss_adv_G = self.bceloss(target_D_out,
                                  torch.FloatTensor(target_D_out.data.size())
                                  .fill_(self.adv_source_label).to(target_D_out.device)) * self.opt.adv
        loss_G = loss_adv_G + loss_GTA
        loss_G.backward()
        self.BaseOpti.step()

        for param in self.net_D.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        source_D_out = self.net_D_DP(F.softmax(source_outputUp.detach(), dim=1))
        target_D_out = self.net_D_DP(F.softmax(target_outputUp.detach(), dim=1))
        loss_D = (
            self.bceloss(source_D_out,
                         torch.FloatTensor(source_D_out.data.size())
                         .fill_(self.adv_source_label).to(source_D_out.device)) +
            self.bceloss(target_D_out,
                         torch.FloatTensor(target_D_out.data.size())
                         .fill_(self.adv_target_label).to(target_D_out.device))
        )
        loss_D.backward()
        self.optimizer_D.step()

        return {
            'src': loss_GTA.item(),
            'G': loss_adv_G.item(),
            'D': loss_D.item()
        }

    def _step_self_training_source(self, source_x, source_label):
        context = torch.cuda.amp.autocast if self.opt.amp else nullcontext

        # ==================================================================================================
        with context():
            ### ---------------------------------- ###
            ### Source: forward with source images ###
            ### ---------------------------------- ###
            source_out = self._forward(self.BaseNet_DP, source_x)
            source_out['out'] = interpb(source_out['out'], source_x.size()[2:])
            loss_src_all = loss_src = cross_entropy2d(inputs=source_out['out'], target=source_label)

            ### ---------------------------------- ###
            ### Source: Cross-domain consistency   ###
            ### ---------------------------------- ###
            loss_src_style = torch.cuda.FloatTensor([0.])
            if self.opt.cross_consist:    # target --> source (style_load)
                forward_from = min(self.opt.style_transfer_layers)
                mid_input = source_out[f'f{forward_from}']
                if self.opt.style_transfer_sg:
                    mid_input = mid_input.detach()  # stop gradient

                src_kwargs = {
                    'label': source_label,
                    'state': 'style_load',
                    'forward_from': forward_from
                }

                features_mean = {}
                features_std = {}
                for l in self.opt.style_transfer_layers:
                    fmean = self.BaseNet.__getattr__(f"features_mean_{l}")  # [n_cls, B, C]
                    fstd = self.BaseNet.__getattr__(f"features_std_{l}")    # [n_cls, B, C]

                    # shuffle
                    perm = torch.randperm(fmean.size(1))
                    fmean = fmean[:, perm, :]
                    fstd = fstd[:, perm, :]

                    # construct new batch axis
                    fmean = torch.stack(fmean.chunk(self.num_devices, 1), dim=0)   # [2, n_cls, B//2, C]
                    fstd = torch.stack(fstd.chunk(self.num_devices, 1), dim=0)     # [2, n_cls, B//2, C]

                    features_mean[l] = fmean
                    features_std[l] = fstd

                src_kwargs.update({
                    'features_mean': features_mean,
                    'features_std': features_std
                })

                source_out2 = self._forward(self.BaseNet_DP, mid_input, **src_kwargs)
                source_out2['out'] = interpb(source_out2['out'], source_x.size()[2:])

                loss_src_style = cross_entropy2d(inputs=source_out2['out'], target=source_label)
                loss_src_all = loss_src_all + loss_src_style

            loss_src_all = loss_src_all * self.opt.srcW

        self._backward(loss_src_all)

        return loss_src, loss_src_style


    def step_self_training(self, source_x, source_label, target_x, target_image2=None, target_pseudo_label=None, 
                           kwargs1={}, kwargs2={}, epoch=None):

        ### Compute increasing factor (gamma) of intra-domain consistency ###
        factor = 1.
        if self.opt.inc_intra_weights == 'linear':
            factor = max(0, (epoch % self.stage_epochs) - self.opt.inc_warmup) / (self.stage_epochs - self.opt.inc_warmup)

        context = torch.cuda.amp.autocast if self.opt.amp else nullcontext

        # ==================================================================================================
        loss_src, loss_src_style = self._step_self_training_source(source_x, source_label)

        # ==================================================================================================
        with context():
            ### Target: Downsample pseudo labels ###
            pseudo_label = interpn(
                target_pseudo_label.unsqueeze(1).float(), scale_factor=0.25).squeeze(1).long()   # [224, 128]
            upsize = pseudo_label.shape[-2:]

            ### Target: forward with target images ###
            tgt_kwargs = {}
            if self.opt.cross_consist:
                tgt_kwargs.update({
                    'label': pseudo_label, 
                    'state': 'style_save'
                })
            target_out = self._forward(self.BaseNet_DP, target_x, **tgt_kwargs)     # OS=8: [113, 66], OS=16: [57, 33]
            target_out['out'] = interpb(target_out['out'], upsize)      # [224, 128]
            target_out['out'] = self.maybe_recover_pos(kwargs1, target_out['out'])

            ### Target: storing target styles ###
            if factor and self.opt.cross_consist:
                for l in self.opt.style_transfer_layers:     # target --> source
                    features_mean = self.BaseNet.__getattr__(f"features_mean_{l}")  # [n_cls, B, C]
                    features_std = self.BaseNet.__getattr__(f"features_std_{l}")
                    memory_ptr = self.BaseNet.__getattr__(f"memory_ptr_{l}")

                    out_mean = target_out[f"means{l}"]          # [?, C]
                    out_std = target_out[f"stds{l}"]            # [?, C]
                    out_classes = target_out[f"classes{l}"]     # [?]
                    
                    for i, c in enumerate(out_classes):
                        ptr = memory_ptr[c]
                        features_mean[c, ptr] = out_mean[i]
                        features_std[c, ptr] = out_std[i]
                        memory_ptr[c] = (ptr + 1) % self.opt.stat_dict_size
            
            loss = torch.cuda.FloatTensor([0.])

            ### ------------------------------------- ###
            ### Target: Intra-domain consistency loss ###
            ### ------------------------------------- ###
            loss_stu_tea = torch.cuda.FloatTensor([0.])
            loss_stu_stu = torch.cuda.FloatTensor([0.])

            if factor and self.opt.intra_consist:
                target_out2 = self._forward(self.BaseNet_DP, target_image2)  # OS=8: [113, 66], OS=16: [57, 33]
                target_out2['out'] = interpb(target_out2['out'], upsize)
                target_out2['out'] = self.maybe_recover_pos(kwargs2, target_out2['out'])

                if self.opt.stu_tea:
                    q1 = interpb(target_out['proj'], upsize)
                    q2 = interpb(target_out2['proj'], upsize)

                    with torch.no_grad():
                        z1 = self._forward(self.BaseNet_ema_DP, target_x)['feat']
                        z1 = interpb(z1, upsize)
                        z1 = self.maybe_recover_pos(kwargs1, z1)
                        z2 = self._forward(self.BaseNet_ema_DP, target_image2)['feat']
                        z2 = interpb(z2, upsize)
                        z2 = self.maybe_recover_pos(kwargs2, z2)
                    
                    loss_stu_tea = self._normalized_l2_loss(q1, q2, z1, z2) * (self.opt.stu_teaW * factor)
                    loss = loss + loss_stu_tea

                if self.opt.stu_stu:
                    out1 = target_out['out']
                    out2 = target_out2['out']

                    loss_stu_stu = self._symmetric_kl_divergence(out1, out2) * (self.opt.stu_stuW * factor)
                    loss = loss + loss_stu_stu

            ### -------------------------- ###
            ### Target: self-training loss ###
            ### -------------------------- ###
            loss_tgt = cross_entropy2d(inputs=target_out['out'], target=pseudo_label)
            
            if factor and self.opt.intra_consist:
                loss_tgt_dual = cross_entropy2d(inputs=target_out2['out'], target=pseudo_label)
                loss_tgt = (loss_tgt + loss_tgt_dual) / 2

            if self.opt.sce:
                rce = self._reverse_cross_entropy(target_out['out'], pseudo_label.clone())
                if factor and self.opt.intra_consist:
                    rce = (rce + self._reverse_cross_entropy(target_out2['out'], pseudo_label.clone())) / 2
                loss_tgt = self.opt.sce_alpha * loss_tgt + self.opt.sce_beta * rce
            loss = loss + loss_tgt

        # ==================================================================================================
        try:
            self._backward(loss)
            self._optimize()
            self.BaseOpti.zero_grad()
        except RuntimeError as e:
            self.logger.error(e)
            self.BaseOpti.zero_grad()
            return {
                'loss_src': 0.,
                'loss_src_style': 0.,
                'loss': 0.,
                'loss_tgt': 0.,
                'loss_stu_tea': 0.,
                'loss_stu_stu': 0.,
            }

        if self.opt.stu_tea:
            q_mm = self.opt.q_momuntum
            with torch.no_grad():
                for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
                    param_k.data = param_k.data * q_mm + param_q.data * (1. - q_mm)
                for (name_q, buffer_q), (name_k, buffer_k) in zip(self.BaseNet.named_buffers(), self.BaseNet_ema.named_buffers()):
                    if "memory_ptr" in name_q or "features_mean" in name_q or "features_std" in name_q:
                        continue
                    buffer_k.data = buffer_k.data * q_mm + buffer_q.data * (1. - q_mm)

        return {
            'src': loss_src.item(),
            'src_style': loss_src_style.item(),
            'loss': loss.item(),
            'tgt': loss_tgt.item(),
            'stu_tea': loss_stu_tea.item(),
            'stu_stu': loss_stu_stu.item(),
        }

    def maybe_recover_pos(self, kwargs, img):
        if kwargs is None or len(kwargs) == 0: return img
        imgs = []
        for i, im in enumerate(img):
            if 'RandomShift' in kwargs and (kwargs['RandomShift'][0][i] > 0 or kwargs['RandomShift'][1][i] > 0):
                im_shift_h = int(kwargs['RandomShift'][0][i] * im.shape[-2])
                im_shift_w = int(kwargs['RandomShift'][1][i] * im.shape[-1])
                im = torch.roll(im, (-im_shift_h, -im_shift_w), (1, 2))
            if 'RandomHorizontallyFlip' in kwargs and kwargs['RandomHorizontallyFlip'][i]:
                im = torch.flip(im, dims=(2,))
            imgs.append(im)
        return torch.stack(imgs, 0)
        
    def _normalized_l2_loss(self, q1, q2, z1, z2):
        q1 = F.normalize(q1, dim=1)
        q2 = F.normalize(q2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # return (-2 * q2 * z1.detach() - 2 * q1 * z2.detach()).sum(1).mean() + 4
        return _normalized_l2_loss_merge(q1, z1.detach(), q2, z2.detach()).sum(1).mean() + 4

    def _symmetric_kl_divergence(self, out1, out2):
        p = torch.log_softmax(out1, dim=1)
        p_tec = torch.softmax(out1, dim=1)
        q = torch.log_softmax(out2, dim=1)
        q_tec = torch.softmax(out2, dim=1)

        kl_loss1 = F.kl_div(p, q_tec, reduction='none').mean(dim=(0, 2, 3)).sum()
        kl_loss2 = F.kl_div(q, p_tec, reduction='none').mean(dim=(0, 2, 3)).sum()

        return kl_loss1 + kl_loss2

    def _reverse_cross_entropy(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != 250).float()
        labels[labels == 250] = self.n_class
        label_one_hot = F.one_hot(labels, self.n_class + 1).float()
        label_one_hot = torch.clamp(label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :], min=1e-4, max=1.0).contiguous()
        sce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return sce

    def freeze_bn_apply(self):
        for net in self.nets_DP:
            net.apply(_freeze_bn)

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def init_device(self, net):
        net = net.to(self.device)
        net = nn.DataParallel(net, device_ids=range(self.num_devices))
        return net
    
    def eval(self, logger=None):
        """Make specific models eval mode during test time"""
        for net in self.nets_DP:
            net.eval()
        if logger is not None:
            logger.info("Successfully set the model eval mode") 
        return

    def train(self):
        for net in self.nets_DP:
            net.train()
        if self.opt.freeze_bn:
            self.freeze_bn_apply()
        return

    def maybe_print_losses(self, accu, timer):
        if self.iter % self.opt.print_interval == 0:
            fmt_str = "[{:d}/{:d}] [{:d}/{:d}] "
            for name in self.loss_names:
                fmt_str += name + ": {:.4f} "
            fmt_str += "[{:.2f}s/c]"
            self.logger.info(fmt_str.format(
                self.epoch, self.opt.epochs, self.iter, self.train_iters, *accu.mean(self.loss_names), timer.spc))

            timer.reset()


@torch.jit.script
def _normalized_l2_loss_merge(q1, z1, q2, z2):
    return -2 * q2 * z1 - 2 * q1 * z2
