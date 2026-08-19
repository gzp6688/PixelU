import math
import torch
import torch.nn as nn
from model_pixelu import PixelU_models
from util.loss import PerceptualLoss

def get_decay_weight(epoch, base_weight=0.01, start_decay=80, end_decay=160):
    if epoch < start_decay:
        return base_weight
    if epoch >= end_decay:
        return 0.0
    progress = (epoch - start_decay) / (end_decay - start_decay)
    decay_factor = 1.0 - progress
    return base_weight * decay_factor

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = PixelU_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.repa_weight = args.repa_weight

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # use x preditction
        self.x_prediction = True

        # perceptual loss (we don't use this in our work)
        self.perceptual_loss_fn = PerceptualLoss(use_lpips=False, use_seg=False)
        self.cached_percept_weight = 1.0

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, zs, epoch, data_iter_step):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        
        e_img = torch.randn_like(x) * self.noise_scale
        z_img = t * x + (1 - t) * e_img
        v_img = (x - z_img) / (1 - t).clamp_min(self.t_eps)

        output_pixel, zs_tilde, zs_tilde_original, cls_logits = self.net(z_img, t.flatten(), labels_dropped)

        # v or x prediction
        if self.x_prediction:
            v_pred_pixel = (output_pixel - z_img) / (1 - t).clamp_min(self.t_eps)
        else:
            v_pred_pixel = output_pixel
        
        # l2 loss
        loss_pixel = (v_img - v_pred_pixel) ** 2
        loss_pixel = loss_pixel.mean(dim=(1, 2, 3)).mean()

        # repa loss
        if zs is not None and zs_tilde is not None:
            # normalize zs and zs_tilde
            zs = torch.nn.functional.normalize(zs, dim=-1) # [B,T,D]
            zs_tilde = torch.nn.functional.normalize(zs_tilde, dim=-1) # [B,T,D]
            # compute cosine similarity
            cos_sim = (zs * zs_tilde).sum(dim=-1)    # [B,T]
            loss_repa = (1 - cos_sim).mean()            
        else:
            loss_repa = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # perc loss
        t_threshold = 0.1  
        lpips_weight = get_decay_weight(epoch, base_weight=0.1, start_decay=80, end_decay=160)
        seg_weight = 0.0
        loss_perc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        mask = t.flatten() >= t_threshold
        if mask.any():
            loss_lpips, loss_seg = self.perceptual_loss_fn(x, output_pixel, mask) 
            loss_perc = lpips_weight * loss_lpips + seg_weight * loss_seg
                
        loss = loss_pixel + (self.repa_weight * loss_repa) + (self.cached_percept_weight * loss_perc)
        return loss, loss_pixel, loss_repa, loss_perc
    

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        out_cond, zs_tilde, zs_tilde_original, cls_logits = self.net(z, t.flatten(), labels)
        if self.x_prediction:
            v_cond = (out_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        else:
            v_cond = out_cond

        # unconditional
        out_uncond, zs_tilde, zs_tilde_original, cls_logits = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        if self.x_prediction:
            v_uncond = (out_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        else:
            v_uncond = out_uncond

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
