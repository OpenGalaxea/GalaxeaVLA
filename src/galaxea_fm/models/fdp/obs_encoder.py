import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import numpy as np
import timm
from typing import Optional, Dict, Tuple, Union, List, Type
from galaxea_fm.utils.pytorch_utils import dict_apply

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class ObsEncoder(nn.Module):
    def __init__(
        self, 
        shape_meta,
        obs_step: int, 
        encoder_model_name: str,
        share_encoder: bool,
        pretrained: bool,
        vit_img_shape: int = 224,
        state_mlp_size=(64, 128), 
        state_mlp_activation_fn=nn.ReLU,
        fusion_mlp_size=(512, 256, 128), 
        fusion_mlp_activation_fn=nn.ReLU,
        additional_convs_channel=(128, 32, 8), 
        additional_convs_kernel_size=[1, 1, 1], 
        additional_convs_stride=[1, 1, 1], 
        additional_convs_padding=[0, 0, 0],
        image_pool_w=10, 
        image_pool_h=8,
        frozen=False,
        use_task_id: bool = False,
        num_tasks: int = 4,
        task_emb_dim: int = 128,
        task_id_key: str = "task_id",
    ):
        super().__init__()
        self.obs_step = obs_step
        self.state_key = 'state'
        self.state_dim = np.sum([meta["shape"] for meta in shape_meta["state"]])
        self.cams_key = [meta["key"] for meta in shape_meta["images"]]
        self.this_encoder = nn.ModuleDict()
        self.share_encoder = share_encoder
        self.encoder_model_name = encoder_model_name

        # ===== ID Embedding  =====
        self.use_task_id = use_task_id
        self.task_id_key = task_id_key
        if self.use_task_id:
            self.task_embed = nn.Embedding(
                num_embeddings=num_tasks,
                embedding_dim=task_emb_dim
            )
            self.task_emb_dim = task_emb_dim
        else:
            self.task_embed = None
            self.task_emb_dim = 0

        # Resnet18 encoder
        if 'resnet' in encoder_model_name:
            resnet18 = torchvision.models.resnet18(pretrained=True)
            resnet18_encoder = nn.Sequential(*list(resnet18.children())[:-2])
            additional_convs = nn.Sequential(
                nn.Conv2d(512, additional_convs_channel[0], kernel_size=additional_convs_kernel_size[0], stride=additional_convs_stride[0], padding=additional_convs_padding[0]),  # Reduce to 256 channels
                nn.ReLU(inplace=True),
                nn.Conv2d(additional_convs_channel[0], additional_convs_channel[1], kernel_size=additional_convs_kernel_size[1], stride=additional_convs_stride[1], padding=additional_convs_padding[1]),  # Reduce to 128 channels
                nn.ReLU(inplace=True),
                nn.Conv2d(additional_convs_channel[1], additional_convs_channel[2], kernel_size=additional_convs_kernel_size[2], stride=additional_convs_stride[2], padding=additional_convs_padding[2]),  # Reduce to 128 channels
                nn.ReLU(inplace=True)
            )
            resnet_encoder = nn.Sequential(
                resnet18_encoder,
                additional_convs
            )
            rgb_encoder = resnet_encoder
            self.out_dim = additional_convs_channel[-1] * len(self.cams_key) * image_pool_w * image_pool_h
        
        # DINO encoder
        elif 'dinov2' in encoder_model_name:
            vit_model = timm.create_model(
                    model_name=encoder_model_name,
                    pretrained=pretrained,
                    global_pool='',         # '' means no pooling
                    num_classes=0,          # remove classification layer
                    img_size=vit_img_shape, # 224
                    drop_path_rate=0.0,     # stochastic depth
                )
            dino_encoder = vit_model
            rgb_encoder = dino_encoder

            if 'vit_small' in encoder_model_name:
                self.out_dim = 384
            elif 'vit_base' in encoder_model_name:
                self.out_dim = 768
            elif 'vit_large' in encoder_model_name:
                self.out_dim = 1024
            else:
                raise NotImplementedError(f"Unknown vit model {encoder_model_name}")
            self.out_dim *= len(self.cams_key)
        elif 'clip' in encoder_model_name:
            rgb_encoder = timm.create_model(
                model_name=encoder_model_name,
                pretrained=pretrained,
                global_pool='', # '' means no pooling
                num_classes=0   # remove classification layer
            )
            if frozen:
                assert pretrained
                for param in rgb_encoder.parameters():
                    param.requires_grad = False
            self.out_dim = 768
            self.out_dim *= len(self.cams_key)
        elif 'siglip' in encoder_model_name:
            rgb_encoder = timm.create_model(
                model_name=encoder_model_name,
                pretrained=pretrained,
                global_pool='', # '' means no pooling
                num_classes=0   # remove classification layer
            )
            if frozen:
                assert pretrained
                for param in rgb_encoder.parameters():
                    param.requires_grad = False
            self.out_dim = 768
            self.out_dim *= len(self.cams_key)

        if share_encoder:
            self.rgb_encoder = rgb_encoder
        else:
            for cam_name in self.cams_key:
                this_model = copy.deepcopy(rgb_encoder)  
                self.this_encoder[cam_name] = this_model

        state_mlp_output_size = 0
        if len(state_mlp_size) == 0:
            self.state_mlp = None
        else:
            if len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            self.state_mlp = nn.Sequential(*create_mlp(self.state_dim, state_mlp_size[-1], net_arch, state_mlp_activation_fn))
            state_mlp_output_size = state_mlp_size[-1]

        fusion_input_dim = self.out_dim * self.obs_step + state_mlp_output_size * self.obs_step
        if self.use_task_id:
            fusion_input_dim += self.task_emb_dim

        if len(fusion_mlp_size) == 0:
            self.fusion_mlp = None
            self.n_output_channels = fusion_input_dim
        else:
            if len(fusion_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = fusion_mlp_size[:-1]
            self.fusion_mlp = nn.Sequential(*create_mlp(fusion_input_dim, fusion_mlp_size[-1], net_arch, fusion_mlp_activation_fn))
            self.n_output_channels = fusion_mlp_size[-1]

    def aggregate_feature(self, feature):
        if 'dinov2' in self.encoder_model_name:
            # vit uses the CLS token
            return feature[:, 0, :]
        elif 'clip' in self.encoder_model_name:
            return feature[:, 0, :]
        elif 'siglip' in self.encoder_model_name:
            return feature[:, 0, :]
        elif 'resnet' in self.encoder_model_name:
            return feature.flatten(1)

    def forward(self, observations: Dict) -> torch.Tensor:
        if not self.share_encoder:
            rgb_feats = []
            for cam_name in self.cams_key:
                rgb = observations[cam_name] # (B, T, 3, H, W)
                B = rgb.shape[0]
                assert self.obs_step == rgb.shape[1]
                rgb = rgb.flatten(0,1)      
                cur = self.this_encoder[cam_name](rgb)
                cur = self.aggregate_feature(cur)
                cur = cur.reshape(B, -1)
                rgb_feats.append(cur)  
            rgb_feats = torch.cat(rgb_feats, dim=-1)
        else:
            rgb_list = []
            cam_num = len(self.cams_key)
            for cam_name in self.cams_key:
                rgb = observations[cam_name] # (B, T, 3, H, W)
                B = rgb.shape[0]
                rgb = rgb.flatten(0,1)
                rgb_list.append(rgb)
            rgb_feat = torch.cat(rgb_list, axis=0) #cbt 3 h w
            rgb_feat = self.rgb_encoder(rgb_feat)
            btc, c, h, w = rgb_feat.shape
            bt = btc // cam_num
            rgb_feat = rgb_feat.reshape(cam_num, bt, c, h, w).permute(1, 0, 2, 3, 4).reshape(bt, -1)
            rgb_feats = rgb_feat.reshape(B,-1)
        feats = [rgb_feats]

        if self.state_mlp is not None:
            state = observations[self.state_key].float() # (B, T, state_dim)
            B = state.shape[0]
            assert self.obs_step == state.shape[1]
            state = state.flatten(0, 1)
            state_feats = self.state_mlp(state)
            state_feats = state_feats.reshape(B, -1)
            feats.append(state_feats)

        # ===== ： ID  nn.Embedding（=1，） =====
        if self.use_task_id:
            task_id = observations[self.task_id_key]  # (B,)
            if task_id.dim() != 1:
                raise ValueError(f"{self.task_id_key} must be 1D (B,), got shape={tuple(task_id.shape)}")
            if task_id.dtype != torch.long:
                task_id = task_id.long()
            task_feats = self.task_embed(task_id)      # (B, task_emb_dim)
            feats.append(task_feats)

        final_feat = torch.cat(feats, dim=-1)
        if self.fusion_mlp is not None:
            final_feat = self.fusion_mlp(final_feat)
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
