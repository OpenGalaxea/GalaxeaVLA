"""
Reference:
- https://github.com/real-stanford/diffusion_policy
"""
import time
import numpy as np
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from galaxea_fm.utils.normalizer import LinearNormalizer
from galaxea_fm.utils.pytorch_utils import dict_apply
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.models.fdp.conditional_unet1d import ConditionalUnet1D


class DiffusionUnetImagePolicy(BasePolicy):
    def __init__(
        self,
        shape_meta, 
        horizon,
        obs_steps, 
        noise_scheduler: Union[DDPMScheduler],
        obs_encoder,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        vision_encoder_lr_scale=0.1, 
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.condition_type = condition_type

        self.action_dim = np.sum([meta["shape"] for meta in shape_meta["action"]])

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.use_rtc = "PiG" in self.noise_scheduler.__class__.__name__
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.obs_steps = obs_steps
        self.vision_encoder_lr_scale = vision_encoder_lr_scale

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        trajectory,
        global_cond,
    ):
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                model_output = self.model(
                    sample=trajectory, timestep=t, local_cond=None, global_cond=global_cond
                )
                # compute previous image: x_t -> x_t-1
                trajectory = self.noise_scheduler.step(
                    model_output, t, trajectory, 
                ).prev_sample
        return trajectory
    
    def conditional_sample_using_rtc(
        self, 
        trajectory: torch.Tensor, 
        global_cond: torch.Tensor, 
        prev_trajectory: torch.Tensor, 
        delay: int, # num of steps of prev traj will be executed due to cur inference delay
        passed: int, # num of steps of prev traj that has already passed
        soft: int, # num of steps of soft mask
    ):
        prev_trajectory = torch.cat(
            [
                prev_trajectory[:, passed:], 
                torch.zeros(trajectory.shape[0], passed, self.action_dim, dtype=self.dtype, device=self.device)
            ], 
            dim=1
        ).flatten(1)
        weights = self.get_rtc_weights(delay, soft)
        weights = weights[None, :, None].repeat(1, 1, self.action_dim).flatten(1)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory = trajectory.clone().requires_grad_(True)
            model_output = self.model(
                sample=trajectory, timestep=t, local_cond=None, global_cond=global_cond, 
            ).flatten(1)
            trajectory = trajectory.flatten(1)
            trajectory = self.noise_scheduler.step(
                model_output=model_output, 
                timestep=t, 
                sample=trajectory, 
                target_sample=prev_trajectory, 
                weights=weights
            ).prev_sample
            trajectory = trajectory.unflatten(1, (self.horizon, self.action_dim))
        
        return trajectory

    def get_rtc_weights(self, delay, soft):
        weights = torch.zeros((self.horizon,), dtype=self.dtype, device=self.device)
        weights[0: delay] = 1
        soft_idx = torch.arange(delay, delay + soft, device=self.device)
        c = (delay + soft - soft_idx - 1) / soft
        weights[soft_idx] = (torch.exp(c) - 1) / (torch.e - 1) * c
        return weights

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        input batch["obs"], output batch["action"]
        """
        # NOTE (junlin): original process is here:
        # batch = self.signal_transform.forward(batch)

        # normalize input
        # nobs = self.normalizer.normalize(batch["obs"], norm_type=self.norm_type)
        # if "prev_action" in batch:
        #     prev_nactions = self.normalizer["action"].normalize(batch["prev_action"], norm_type=self.norm_type)
        # ==================================================

        # NOTE (junlin): process the data with FDPProcessor
        nobs = {}

        images = batch["pixel_values"]
        # NOTE (junlin): camera order followed the shape_meta from yaml config.
        for i, meta in enumerate(self.shape_meta["images"]):
            nobs[meta["key"]] = images[:, i: i + 1, :, :, :]

        nobs["state"] = batch["proprio"]

        if "prev_action" in batch:
            prev_nactions = batch["prev_action"]
        
        if "task_id" in batch:
            nobs["task_id"] = batch["task_id"]

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        nobs_features = self.obs_encoder(nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # empty data for action
        trajectory = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            device=global_cond.device, 
            dtype=global_cond.dtype
        )

        # run sampling
        if self.use_rtc:
            nsample = self.conditional_sample_using_rtc(
                trajectory=trajectory, 
                global_cond=global_cond, 
                prev_trajectory=prev_nactions, 
                delay=batch["delay"], 
                passed=batch["passed"], 
                soft=batch["soft"], 
            )
        else:
            nsample = self.conditional_sample(
                trajectory,
                global_cond=global_cond,
            )
        batch["action"] = nsample

        return batch

    def compute_loss(self, batch):
        # NOTE (junlin): process the data with FDPProcessor
        nobs = {}

        images = batch["pixel_values"]
        # NOTE (junlin): camera order followed the shape_meta from yaml config.
        for i, meta in enumerate(self.shape_meta["images"]):
            nobs[meta["key"]] = images[:, i : i + self.obs_steps, ...]

        nobs["state"] = batch["proprio"]

        if "task_id" in batch:
            nobs["task_id"] = batch["task_id"]

        nobs = nobs
        nactions = batch["action"]
        # ==================================================

        nobs_features = self.obs_encoder(nobs)

        # Sample noise that we'll add to the images
        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # Predict the noise residual
        pred = self.model(
            sample=noisy_trajectory.float(), 
            timestep=timesteps, 
            local_cond=None, 
            global_cond=nobs_features
        ) # [batch_size, horizon, action_dim]

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss_log = dict()

        dim_loss_keys = [f"train_diffuse_loss/dim_{i:02d}" for i in range(loss.shape[2])]
        dim_loss_vals = [i for i in loss.detach().mean(dim=(0, 1))]
        loss_log.update(dict(zip(dim_loss_keys, dim_loss_vals)))
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_log["train/diffuse_loss"] = loss.detach()
        
        return loss , loss_log

    def get_optim_param_groups(self, lr, weight_decay):
        other_params, vision_encoder_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("obs_encoder.vision_encoders"):
                    vision_encoder_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {
                "params": vision_encoder_params, 
                "lr": lr * self.vision_encoder_lr_scale, 
                "weight_decay": weight_decay, 
                "name": "vision_encoder", 
            }, 
            {
                "params": other_params, 
                "lr": lr, 
                "weight_decay": weight_decay, 
                "name": "diffusion_model", 
            },

        ]
        return param_groups