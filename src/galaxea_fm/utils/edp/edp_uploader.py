import logging
import os
from pathlib import Path

import tos
import torch
from git import Repo
from tos.utils import SizeAdapter
from omegaconf import DictConfig

from galaxea_fm.utils.edp import utils
from galaxea_fm.utils.edp.user import EDP_USER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             retries = 0
#             while retries < max_retries:
#                 try:
#                     return func(*args, **kwargs)
#                 except exceptions as e:
#                     logger.warning(f"Exception occurred: {e}. Retrying ({retries + 1}/{max_retries})...")
#                     retries += 1
#                     time.sleep(delay)
#             logger.error("Max retries reached. Operation failed.")
#             return None
#         return wrapper
#     return decorator


class EDPCardCreator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.has_card = self.cfg.card is not None
        repo = Repo(".")
        self.git_branch = repo.active_branch.name
        self.git_commit = repo.head.commit.hexsha

    def extract_info(self):
        # dataID
        dataID = []
        try:
            model_server_path = self.cfg["save_dir"]
            # training_dataset_root = self.cfg['root']
            # training_dataset_repo_ids = self.cfg['repo_ids']
            # training_dataset_num = 0
            # # 
            # for search_path in training_dataset_dirs:
            #     search_path = search_path.replace("mnt_0", "mnt_1")
            #     training_dataset_num += len([
            #         folder for folder in os.listdir(search_path)
            #         if os.path.isdir(os.path.join(search_path, folder))
            #         and folder.endswith("RAW-000")
            #     ])
            #     json_file_path = os.path.dirname(search_path) + '/training_data_set_meta.json'
            #     if not os.path.exists(json_file_path):
            #         print(f"Warning: File {json_file_path} does not exist.")
            #         continue
            #     # 
            #     with open(json_file_path, 'r', encoding='utf-8') as f:
            #         try:
            #             data = json.load(f)
            #         except json.JSONDecodeError:
            #             print("Warning: Failed to parse JSON file.")
            #             continue
            #     if 'data' in data:
            #         data_id = data['data'].get('trainingDataSetId')
            #     else:
            #         data_id = data.get('trainingDataSetId')
            #     dataID.append(data_id)
            # ID
            if len(dataID) < 1:
                dataID.append(33)

            batch_size = self.cfg["batch_size"]

            tags = self.cfg["tags"]
            # 
            traningtime_raw = self.cfg["training_time"]
            traningtime = traningtime_raw.replace("_", " ")
            traningtime_ = traningtime.split(" ", 1)
            date_part, time_part = traningtime_
            time_part = time_part.replace("-", ":")
            traningtime = f"{date_part} {time_part}"
            # 

            edp_info = {
                "name": self.cfg["card"],
                "trainedBy": EDP_USER,
                "trainingTime": traningtime,
                "git_branch": self.git_branch,
                "git_commit": self.git_commit,
                "trainingDataSetIds": dataID,
                "tags": tags,
                "description": {
                    # 'dataset_num': training_dataset_num,
                    "path": model_server_path,
                    "batch_size": batch_size,
                },
                "step": self.cfg["max_steps"],
            }
            return edp_info

        except Exception as e:
            logger.error(f"Error extracting model info: {e}", exc_info=True)
            raise

    # @retry_on_exception(max_retries=3, delay=2, exceptions=(ConnectionError, TimeoutError))
    def set_card_model(self, edp_info_dict):

        name = edp_info_dict["name"]
        description = str(edp_info_dict["description"])
        trainedBy = edp_info_dict["trainedBy"]
        trainingTime = edp_info_dict["trainingTime"]
        gitBranch = edp_info_dict["git_branch"]
        gitCommit = edp_info_dict["git_commit"]
        trainingDataSetIds = edp_info_dict["trainingDataSetIds"]
        step = edp_info_dict["step"]
        card_response = utils.create_model_card(
            "project-s",
            name,
            description,
            trainedBy,
            trainingTime,
            gitBranch,
            gitCommit,
            trainingDataSetIds,
        )
        des = ""
        if card_response["data"] is None:
            card_response = utils.get_model_card_by_name(name)
            if card_response["data"] is None:
                print(f"model card {name} not found")
                return None
            card_id = card_response["data"]["records"][0]["id"]
            print(f"model card {name} already exists, card_id: {card_id}")
        else:
            card_id = card_response["data"]
        model_response = utils.create_model(card_id, des, step, "pt")
        oss = utils.get_model(model_response["data"])
        return oss

    def upload(self):
        ak = os.getenv("VOLC_AK", "")
        sk = os.getenv("VOLC_SK", "")
        bucket_name = "edp"
        endpoint = "tos-cn-beijing.volces.com"
        region = "cn-beijing"
        client = tos.TosClientV2(ak, sk, endpoint=endpoint, region=region)

        # pt
        files_root_path = self.cfg["save_dir"]
        model_path = files_root_path + "/checkpoints/last.pt"
        config_yaml_path = files_root_path + "/.hydra/config.yaml"
        model_path = str(Path(model_path).resolve())
        config_yaml_path = str(Path(config_yaml_path).resolve())
        ckpt = torch.load(model_path, weights_only=False)
        model_state_dict = {"model_state_dict": ckpt["model_state_dict"]}
        #  model_state_dict  .pt 
        model_pt = "model_state_dict.pt"
        torch.save(model_state_dict, model_pt)
        print(f"model_state_dict  {model_pt}")
        model_cfg = config_yaml_path

        edp_info_dict = self.extract_info()
        res = self.set_card_model(edp_info_dict)
        object_key = res["data"]["records"][0]["ossDir"]

        total_size = os.path.getsize(model_pt)
        part_size = 100 * 1024 * 1024
        try:
            #  TosClientV2 ， TosClientV2 
            client = tos.TosClientV2(ak, sk, endpoint, region)
            # config.yaml
            result = client.put_object_from_file(
                bucket_name, object_key + "/config.yaml", model_cfg
            )
            # HTTP
            print("http status code:{}".format(result.status_code))
            # ID。ID，
            print("request_id: {}".format(result.request_id))
            # hash_crc64_ecma 64CRC, 
            print("crc64: {}".format(result.hash_crc64_ecma))
            # ，storage_class
            # ACL，acl、grant_full_control
            multi_result = client.create_multipart_upload(
                bucket_name,
                object_key + "/model_state_dict.pt",
                acl=tos.ACLType.ACL_Public_Read,
                storage_class=tos.StorageClassType.Storage_Class_Standard,
            )

            upload_id = multi_result.upload_id
            parts = []

            # 
            with open(model_pt, "rb") as f:
                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    out = client.upload_part(
                        bucket_name,
                        object_key + "/model_state_dict.pt",
                        upload_id,
                        part_number,
                        content=SizeAdapter(f, num_to_upload, init_offset=offset),
                    )
                    parts.append(out)
                    offset += num_to_upload
                    part_number += 1

            # 
            client.complete_multipart_upload(
                bucket_name, object_key + "/model_state_dict.pt", upload_id, parts
            )

            if os.path.exists(model_pt):
                os.remove(model_pt)
                print(f": {model_pt}")
        except tos.exceptions.TosClientError as e:
            # ，，
            print(
                "fail with client error, message:{}, cause: {}".format(
                    e.message, e.cause
                )
            )
        except tos.exceptions.TosServerError as e:
            # ，，
            print("fail with server error, code: {}".format(e.code))
            # request id ，
            print("error with request id: {}".format(e.request_id))
            print("error with message: {}".format(e.message))
            print("error with http code: {}".format(e.status_code))
            print("error with ec: {}".format(e.ec))
            print("error with request url: {}".format(e.request_url))
        except Exception as e:
            print("fail with unknown error: {}".format(e))
