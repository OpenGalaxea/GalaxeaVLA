import os
import re
import subprocess
from types import MethodType

import torch
import yaml
import hydra
import numpy as np
import onnx
import onnxoptimizer as optimizer
from hydra.core.global_hydra import GlobalHydra
from accelerate import PartialState
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from polygraphy.backend.onnx import fold_constants

from galaxea_fm.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from galaxea_fm.models.fdp.unet_policy import DiffusionUnetImagePolicy
from galaxea_fm.utils.edp.edp_client import EDPClient


"""
Export model to ONNX format and upload to EDP platform
use:
python export_onnx.py --model_card <model_card_name>
"""

logger = get_logger(__name__)


class ONNXExporter:
    def __init__(self, model_card=None, cfg=None):
        self.edp_client = EDPClient()
        if cfg is not None:
            self.cfg = cfg
            self.output_dir = cfg.output_dir
            self.model_path = self.output_dir + "/checkpoints/last.pt"
            self.use_model_card = False
            self.model_name = cfg.edp.card
        else:
            self.model_card = model_card
            self.model_name = self.model_card
            self.use_model_card = True
            self.output_dir = os.getcwd()
            self.model_path = (
                self.output_dir + f"/{self.model_name}/model_state_dict.pt"
            )
        output_dir = os.path.join(self.output_dir, f"{self.model_name}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.encoder_onnx_file = os.path.join(output_dir, f"encoder.onnx")
        self.predictor_onnx_file = os.path.join(output_dir, f"predictor.onnx")
        self.model_card_yaml_file = os.path.join(
            self.output_dir, f"{self.model_name}/model_card.yaml"
        )

    def export(self):
        if self.use_model_card and not os.path.exists(self.model_path):
            print(f"downloading model from edp")
            self.edp_client.get_pytorch_model_by_name(self.model_name)
            print(f"downloading model from edp success")
        else:
            self.edp_client.get_model_card_id(self.model_name)
        self._load_model()

        self.generate_model_card_yaml()

        # encoder
        encoder = self.policy.obs_encoder
        obs_steps = self.cfg.model.model_arch.obs_steps
        shape_meta = self.cfg.model.model_arch.shape_meta
        state_shape = 0
        for state_meta in shape_meta.state:
            state_shape += state_meta.shape

        image_metas = {}
        for image_meta in shape_meta.images:
            image_metas[image_meta.key] = image_meta

        state_tensor = torch.randn(
            1, obs_steps, state_shape, dtype=torch.float64
        ).cuda()
        image_tensor_dict = {}
        for key, obs_cfg in image_metas.items():
            image_tensor_dict[key] = torch.randn(
                1,
                obs_steps,
                obs_cfg.shape[0],
                obs_cfg.shape[1],
                obs_cfg.shape[2],
                dtype=torch.float32,
            ).cuda()
        task_id_tensor = torch.tensor([0], dtype=torch.int32).cuda()

        input_names = [
            "state",
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
            "task_id",
        ]
        output_names = ["nobs_features"]
        original_forward = encoder.forward

        def forward(
            self,
            state_tensor,
            head_rgb_tensor,
            left_wrist_rgb_tensor,
            right_wrist_rgb_tensor,
            task_id_tensor,
        ):
            observations = {
                "state": state_tensor,
                "head_rgb": head_rgb_tensor,
                "left_wrist_rgb": left_wrist_rgb_tensor,
                "right_wrist_rgb": right_wrist_rgb_tensor,
                "task_id": task_id_tensor,
            }
            nobs_features = original_forward(observations)
            print(f"nobs_features shape: {nobs_features.shape}")
            feats = {
                "final_feat": nobs_features,
            }
            return feats

        encoder.forward = MethodType(forward, encoder)

        torch.onnx.export(
            encoder,
            (
                state_tensor,
                image_tensor_dict["head_rgb"],
                image_tensor_dict["left_wrist_rgb"],
                image_tensor_dict["right_wrist_rgb"],
                task_id_tensor,
            ),
            self.encoder_onnx_file,
            verbose=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )

        predictor = self.policy.model
        action_shape = 0
        for action_meta in shape_meta.action:
            action_shape += action_meta.shape
        action_size = self.cfg.data.action_size
        sample = torch.randn(1, action_size, action_shape, dtype=torch.float32).to(
            "cuda"
        )
        # TODO: fix hard-code
        timestep = torch.tensor([1], dtype=torch.int32).to("cuda")
        global_cond = torch.randn(1, 512, dtype=torch.float32).to("cuda")
        local_cond = None
        input_names = ["sample", "timestep", "global_cond", "local_cond"]
        output_names = ["latent"]

        torch.onnx.export(
            predictor,
            (sample, timestep, local_cond, global_cond),
            self.predictor_onnx_file,
            verbose=True,
            opset_version=15,
            do_constant_folding=True,  # 
            input_names=input_names,
            output_names=output_names,
        )

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

        all_passes = optimizer.get_available_passes()
        fusion_passes = [
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_add_bias_into_conv",
            "constant_folding",
            "eliminate_deadend",
            "eliminate_nop_pad",
            "nchw2nhwc",
            "nhwc2nchw",
        ]
        supported_passes = [p for p in fusion_passes if p in all_passes]

        encoder_onnx = fold_constants(onnx.load(self.encoder_onnx_file))
        optimized_encoder = optimizer.optimize(
            encoder_onnx,
            passes=supported_passes,
        )
        from onnx import version_converter

        optimized_encoder.ir_version = 10
        optimized_encoder = version_converter.convert_version(optimized_encoder, 15)
        onnx.save(optimized_encoder, self.encoder_onnx_file)

        predictor_onnx = fold_constants(onnx.load(self.predictor_onnx_file))
        optimized_predictor = optimizer.optimize(
            predictor_onnx, passes=supported_passes
        )
        optimized_predictor.ir_version = 10
        optimized_predictor = version_converter.convert_version(optimized_predictor, 15)
        onnx.save(optimized_predictor, self.predictor_onnx_file)

        process = subprocess.Popen(
            ["python", "-m", "onnxsim", self.encoder_onnx_file, self.encoder_onnx_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
        if process.returncode == 0:
            print(f"encoder export success")

        process = subprocess.Popen(
            [
                "python",
                "-m",
                "onnxsim",
                self.predictor_onnx_file,
                self.predictor_onnx_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
        if process.returncode == 0:
            print(f"predictor export success")

    def generate_model_card_yaml(
        self,
    ):
        model_cards = dict()

        #  normalizer （“” normalizer， tid ）
        state_scale = (
            self.policy.normalizer["state"].params_dict["scale"].cpu().tolist()
        )
        state_offset = (
            self.policy.normalizer["state"].params_dict["offset"].cpu().tolist()
        )
        action_scale = (
            self.policy.normalizer["action"].params_dict["scale"].cpu().tolist()
        )
        action_offset = (
            self.policy.normalizer["action"].params_dict["offset"].cpu().tolist()
        )

        def to_list(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            return list(x)

        def to_scalar(x):
            arr = np.array(x)
            if arr.ndim == 0:
                return float(arr)
            return float(arr.reshape(-1)[0])

        is_multitask = (
            getattr(self.dataset, "dataset_batch_to_task_id", None) is not None
            and len(getattr(self.dataset, "dataset_batch_to_task_id", {})) > 0
        )

        if is_multitask:
            # （ model_name  taskXXX； model_name）
            m = re.search(r"task\d+", self.model_name)
            if m:
                task_prefix = m.group()
            else:
                task_prefix = None
                for ep in self.dataset.episode_paths:
                    root = str((ep.parent).parent)
                    m2 = re.search(r"task\d+", root)
                    if m2:
                        task_prefix = m2.group()
                        break
                if task_prefix is None:
                    task_prefix = self.model_name

            #  task_id
            unique_roots = {}
            for ep in self.dataset.episode_paths:
                root = str((ep.parent).parent)
                if root not in unique_roots:
                    tid = self.dataset.dataset_batch_to_task_id.get(root, None)
                    if tid is None:
                        raise ValueError(f"task_id not found for dataset batch: {root}")
                    unique_roots[root] = int(tid)

            for root, tid in unique_roots.items():
                sm = re.search(r"box\d+", root)
                sub_key = sm.group() if sm else f"id{tid}"
                card_key = f"{task_prefix}-{sub_key}"

                init_pos = self.dataset.get_init_positions(task_id=tid)
                for key_req in [
                    "left_arm",
                    "right_arm",
                    "left_gripper",
                    "right_gripper",
                ]:
                    if key_req not in init_pos:
                        raise ValueError(
                            f"init positions missing key: {key_req} for task id {tid}"
                        )

                init_pose = {
                    "left_arm_init_position": to_list(
                        np.array(init_pos["left_arm"])[:7]
                    ),
                    "right_arm_init_position": to_list(
                        np.array(init_pos["right_arm"])[:7]
                    ),
                    "left_gripper_position": to_scalar(init_pos["left_gripper"]),
                    "right_gripper_position": to_scalar(init_pos["right_gripper"]),
                    "torso_position": [0.0, 0.0, 0.0, 0.0],
                    "use_left_arm": True,
                    "use_right_arm": True,
                }

                model_cards[card_key] = {
                    "state_scale": list(state_scale),
                    "state_offset": list(state_offset),
                    "action_scale": list(action_scale),
                    "action_offset": list(action_offset),
                    "init_pose": init_pose,
                    "init_preload": False,
                    "encoder_path": f"{self.model_name}/encoder_fp16.plan",
                    "predictor_path": f"{self.model_name}/predictor_fp16.plan",
                    "ensemble_mode": "RTG",
                    "action_step": 32,
                    "k_act": 0.01,
                    "tau_hato": 0.5,
                    "id": tid,
                    "multi_task_id": tid,
                    "use_slerp_quat_avg": True,
                    "elapsed_time": 30,
                    "val_transforms": OmegaConf.to_container(
                        self.cfg.model.model_arch.val_transforms, resolve=True
                    ),
                    "shape_meta": OmegaConf.to_container(
                        self.cfg.model.model_arch.shape_meta, resolve=True
                    ),
                }
        else:
            m = re.search(r"task\d+", self.model_name)
            if not m:
                raise ValueError(
                    f'model_card "{self.model_name}" does not contain pattern task\\d+'
                )
            task_name = m.group()

            model_cards[self.model_name] = {
                "state_scale": list(state_scale),
                "state_offset": list(state_offset),
                "action_scale": list(action_scale),
                "action_offset": list(action_offset),
                "init_pose": {},
                "init_preload": False,
                "encoder_path": f"{self.model_name}/encoder_fp16.plan",
                "predictor_path": f"{self.model_name}/predictor_fp16.plan",
                "ensemble_mode": "RTG",
                "action_step": 32,
                "k_act": 0.01,
                "tau_hato": 0.5,
                "use_slerp_quat_avg": True,
                "elapsed_time": 30,
                "val_transforms": OmegaConf.to_container(
                    self.cfg.model.model_arch.val_transforms, resolve=True
                ),
                "shape_meta": OmegaConf.to_container(
                    self.cfg.model.model_arch.shape_meta, resolve=True
                ),
            }

            model_cards[self.model_name]["id"] = int(task_name[4:])
            model_cards[self.model_name]["multi_task_id"] = model_cards[
                self.model_name
            ]["id"]

            init_positions = self.dataset.get_init_positions()

            model_cards[self.model_name]["init_pose"]["left_arm_init_position"] = (
                init_positions["left_arm"][0:7].tolist()
            )
            model_cards[self.model_name]["init_pose"]["right_arm_init_position"] = (
                init_positions["right_arm"][0:7].tolist()
            )
            model_cards[self.model_name]["init_pose"]["left_gripper_position"] = (
                init_positions["left_gripper"].tolist()
            )
            model_cards[self.model_name]["init_pose"]["right_gripper_position"] = (
                init_positions["right_gripper"].tolist()
            )
            model_cards[self.model_name]["init_pose"]["use_left_arm"] = True
            model_cards[self.model_name]["init_pose"]["use_right_arm"] = True
            model_cards[self.model_name]["val_transforms"] = OmegaConf.to_container(
                self.cfg.model.model_arch.val_transforms, resolve=True
            )
            model_cards[self.model_name]["shape_meta"] = OmegaConf.to_container(
                self.cfg.model.model_arch.shape_meta, resolve=True
            )

            model_cards[self.model_name]["init_pose"]["use_torso"] = False
            for meta in self.cfg.model.model_arch.shape_meta.action:
                if "torso" in meta.key:
                    model_cards[self.model_name]["init_pose"]["use_torso"] = True

            if model_cards[self.model_name]["init_pose"]["use_torso"]:
                model_cards[self.model_name]["init_pose"]["torso_position"] = (
                    init_positions["torso"].tolist()
                )
            else:
                model_cards[self.model_name]["init_pose"]["torso_position"] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

        payload = {"model_cards": model_cards}

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        def custom_list_representer(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, custom_list_representer, Dumper=NoAliasDumper)
        with open(self.model_card_yaml_file, "w") as file:
            yaml.dump(payload, file, sort_keys=False, Dumper=NoAliasDumper)

    def upload(
        self,
    ):
        self.edp_client.upload_onnx(self.model_name, self.encoder_onnx_file)
        self.edp_client.upload_onnx(self.model_name, self.predictor_onnx_file)
        self.edp_client.upload_yaml(self.model_card_yaml_file)

    def notify(
        self,
    ):
        self.edp_client.notify_compile_server()

    def _load_model(self):
        if self.use_model_card:
            GlobalHydra.instance().clear()
            hydra.initialize(version_base="1.3", config_path=self.model_card)
            self.cfg = hydra.compose(
                config_name="config.yaml",
                overrides=[],  # ，
            )
        # if self.cfg.get("seed"):
        # L.seed_everything(self.cfg.seed, workers=True)
        # TODO seed set
        self.model = hydra.utils.instantiate(self.cfg.model.model_arch)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location="cuda:0", weights_only=False)[
                "model_state_dict"
            ]
        )
        self.policy = self.model.cuda().eval()
        self.dataset: GalaxeaLerobotDataset = hydra.utils.instantiate(self.cfg.data)

    def export_onnx(
        self,
    ):
        self.export()
        self.upload()
        self.notify()


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", required=True, type=str, help="model card name")

    return parser.parse_args()


if __name__ == "__main__":
    distributed_state = PartialState()
    args = get_args()
    model_card = args.model_card
    onnx_exporter = ONNXExporter(model_card=model_card)
    onnx_exporter.export()
    onnx_exporter.upload()
    onnx_exporter.notify()
