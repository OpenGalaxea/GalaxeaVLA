# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Literal, List
import torch
import torch.nn.functional as F

from galaxea_fm.utils.rotation import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    rotation_6d_to_matrix, 
    matrix_to_rotation_9d, 
    rotation_9d_to_matrix, 
    quaternion_to_axis_angle, 
    axis_angle_to_quaternion, 
)

class PoseRotationTransform:
    def __init__(self, rotation_type: Literal["quaternion", "rotation_6d", "rotation_9d"], category_keys: List[str]):
        self.rotation_type = rotation_type
        self.category_keys = category_keys

    def forward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat == "action" and "action" not in batch:
                continue

            for k in ks:
                batch[cat][k] = self._forward(batch[cat][k])

        return batch
    
    def backward(self, batch):
        for cat, ks in self.category_keys.items():
            for k in ks:
                batch[cat][k] = self._backward(batch[cat][k])

        return batch


    def _forward(self, pose):
        assert pose.shape[-1] == 7
        if self.rotation_type == "quaternion":
            return pose
        else: 
            position = pose[..., 0: 3]
            quaternion = pose[..., [6, 3, 4, 5]]
            matrix = quaternion_to_matrix(quaternion)
            if self.rotation_type == "rotation_6d":
                rotation = matrix_to_rotation_6d(matrix)
            elif self.rotation_type == "rotation_9d":
                rotation = matrix_to_rotation_9d(matrix)
            else:
                raise NotImplementedError
            return torch.cat([position, rotation], axis=-1)
        
    def _backward(self, pose: torch.Tensor):
        if self.rotation_type == "quaternion":
            return pose
        else:
            position = pose[..., 0: 3]
            if self.rotation_type == "rotation_6d":
                assert pose[..., 3:].shape[-1] == 6
                matrix = rotation_6d_to_matrix(pose[..., 3:])
            elif self.rotation_type == "rotation_9d":
                assert pose[..., 3:].shape[-1] == 9
                matrix = rotation_9d_to_matrix(pose[..., 3:])
            else:
                raise NotImplementedError
            quaternion = matrix_to_quaternion(matrix)
            quaternion = quaternion[..., [1, 2, 3, 0]]
            return torch.cat([position, quaternion], axis=-1)

    def add_noise(self, pose: torch.Tensor, std_position=0.05, std_angle=0.05):
        assert pose.shape[-1] == 7
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]]
        axis_angles = quaternion_to_axis_angle(quaternion)
        position = position + std_position * torch.randn_like(position)
        axis_angles = axis_angles + std_angle * torch.randn_like(axis_angles)
        quaternion = axis_angle_to_quaternion(axis_angles)
        quaternion = quaternion[..., [1, 2, 3, 0]]
        return torch.cat([position, quaternion], axis=-1)
    

