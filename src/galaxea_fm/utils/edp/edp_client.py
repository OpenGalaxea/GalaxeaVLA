import json
import time
import os
import hashlib
import tos
import requests
import yaml

from pathlib import Path
from tqdm import tqdm
from tos import DataTransferType
from tos.utils import MergeProcess, SizeAdapter

EDP_AK = os.environ.get("EDP_AK")
EDP_SK = os.environ.get("EDP_SK")
TOS_AK = os.environ.get("TOS_AK")
TOS_SK = os.environ.get("TOS_SK")
BUCKET_NAME = "edp"
ENDPOINT = "tos-cn-beijing.volces.com"
REGION = "cn-beijing"


class EDPClient:
    def __init__(
        self,
    ):
        self.tos_client = tos.TosClientV2(TOS_AK, TOS_SK, ENDPOINT, REGION)
        self.root_dir = os.getcwd()

    def download(self, remote_file, local_file):
        object_stream = self.tos_client.get_object_to_file(
            BUCKET_NAME, remote_file, local_file
        )
        try:
            for content in object_stream:
                print(content)
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

    def upload_yaml(self, file_path):
        file_name = os.path.basename(file_path)
        object_key = f"{self.onnx_model_oss_dir}/{file_name}"
        with open(file_path, "r", encoding="utf-8") as file:
            yaml_content = file.read()
        resp = self.tos_client.put_object(BUCKET_NAME, object_key, content=yaml_content)
        assert resp.status_code == 200

    def upload_report(self, file_path, oss_path):
        file_name = os.path.basename(file_path)
        object_key = f"{oss_path}/{file_name}"
        with open(file_path, "r", encoding="utf-8") as file:
            yaml_content = file.read()
        resp = self.tos_client.put_object(BUCKET_NAME, object_key, content=yaml_content)
        assert resp.status_code == 200

    def upload_onnx(self, model_card, file_path):
        file_name = os.path.basename(file_path)
        total_size = os.path.getsize(file_path)
        part_size = 5 * 1024 * 1024

        model_card_id = self._get_model_card_by_name(model_card)["data"]["records"][0][
            "id"
        ]
        self._create_model(int(model_card_id), "", "20000", "onnx")
        models_info = self._get_model_by_model_card_id(model_card_id)["data"]["records"]
        self.onnx_model_oss_dir = self._get_model_oss_dir(models_info, "onnx")
        desc = f"Uploading {file_path}"
        object_key = f"{self.onnx_model_oss_dir}/{file_name}"
        with tqdm(total=100, desc=desc) as pbar:

            def percentage(
                consumed_bytes: int,
                total_bytes: int,
                rw_once_bytes: int,
                type: DataTransferType,
            ):
                percent = int(100 * float(consumed_bytes) / float(total_bytes))
                pbar.update(percent - pbar.n)

            data_transfer_listener = MergeProcess(
                percentage, total_size, (total_size + part_size - 1) // part_size, 0
            )

            multi_result = self.tos_client.create_multipart_upload(
                BUCKET_NAME, object_key
            )

            upload_id = multi_result.upload_id
            parts = []

            with open(file_path, "rb") as f:
                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    out = self.tos_client.upload_part(
                        BUCKET_NAME,
                        object_key,
                        upload_id,
                        part_number,
                        content=SizeAdapter(f, num_to_upload, init_offset=offset),
                        data_transfer_listener=data_transfer_listener,
                    )
                    parts.append(out)
                    offset += num_to_upload
                    part_number += 1

            self.tos_client.complete_multipart_upload(
                BUCKET_NAME, object_key, upload_id, parts
            )

    def upload_tensorrt_plan(self, file_path, oss_dir):
        file_name = os.path.basename(file_path)
        total_size = os.path.getsize(file_path)
        part_size = 5 * 1024 * 1024

        desc = f"Uploading {file_path}"
        object_key = f"{oss_dir}/{file_name}"
        with tqdm(total=100, desc=desc) as pbar:

            def percentage(
                consumed_bytes: int,
                total_bytes: int,
                rw_once_bytes: int,
                type: DataTransferType,
            ):
                percent = int(100 * float(consumed_bytes) / float(total_bytes))
                pbar.update(percent - pbar.n)

            data_transfer_listener = MergeProcess(
                percentage, total_size, (total_size + part_size - 1) // part_size, 0
            )

            multi_result = self.tos_client.create_multipart_upload(
                BUCKET_NAME, object_key
            )

            upload_id = multi_result.upload_id
            parts = []

            with open(file_path, "rb") as f:
                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    out = self.tos_client.upload_part(
                        BUCKET_NAME,
                        object_key,
                        upload_id,
                        part_number,
                        content=SizeAdapter(f, num_to_upload, init_offset=offset),
                        data_transfer_listener=data_transfer_listener,
                    )
                    parts.append(out)
                    offset += num_to_upload
                    part_number += 1

            self.tos_client.complete_multipart_upload(
                BUCKET_NAME, object_key, upload_id, parts
            )

    def get_model_card_id(self, model_card):
        model_card_id = self._get_model_card_by_name(model_card)["data"]["records"][0][
            "id"
        ]
        self.model_card_id = model_card_id

    def get_pytorch_model_by_name(self, model_card):
        model_card_id = self._get_model_card_by_name(model_card)["data"]["records"][0][
            "id"
        ]
        self.model_card_id = model_card_id
        models_info = self._get_model_by_model_card_id(model_card_id)["data"]["records"]
        pt_model_oss_dir = self._get_model_oss_dir(models_info, "pt")
        # plan_fp16
        # onnx
        os.makedirs(os.path.join(self.root_dir, model_card), exist_ok=True)
        config_key = pt_model_oss_dir + "/config.yaml"
        pt_model_key = pt_model_oss_dir + "/model_state_dict.pt"

        config_file = os.path.join(self.root_dir, f"{model_card}/config.yaml")
        pt_model_file = os.path.join(self.root_dir, f"{model_card}/model_state_dict.pt")
        self.download(config_key, config_file)
        self.download(pt_model_key, pt_model_file)

        with Path(config_file).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["ckpt_path"] = pt_model_file

        with Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(data, f)

    def get_model_by_name(self, model_card, model_type):
        model_card_id = self._get_model_card_by_name(model_card)["data"]["records"][0][
            "id"
        ]
        self.model_card_id = model_card_id
        models_info = self._get_model_by_model_card_id(model_card_id)["data"]["records"]
        if model_type == "plan_fp16":
            model_oss_dir = self._get_model_oss_dir(models_info, "plan_fp16")
        elif model_type == "onnx":
            model_oss_dir = self._get_model_oss_dir(models_info, "onnx")
        elif model_type == "pt":
            model_oss_dir = self._get_model_oss_dir(models_info, "pt")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        os.makedirs(os.path.join(self.root_dir, model_card), exist_ok=True)

        online_path = []
        local_path = []
        if model_type == "pt":
            online_path.append(model_oss_dir + "/config.yaml")
            online_path.append(model_oss_dir + "/model_state_dict.pt")
            local_path.append(os.path.join(self.root_dir, f"{model_card}/config.yaml"))
            local_path.append(
                os.path.join(self.root_dir, f"{model_card}/model_state_dict.pt")
            )
        elif model_type == "onnx":
            online_path.append(model_oss_dir + "/encoder.onnx")
            online_path.append(model_oss_dir + "/predictor.onnx")
            online_path.append(model_oss_dir + "/model_card.yaml")
            local_path.append(os.path.join(self.root_dir, f"{model_card}/encoder.onnx"))
            local_path.append(
                os.path.join(self.root_dir, f"{model_card}/predictor.onnx")
            )
            local_path.append(
                os.path.join(self.root_dir, f"{model_card}/model_card.yaml")
            )
        elif model_type == "plan_fp16":
            online_path.append(model_oss_dir + "/encoder_fp16.plan")
            online_path.append(model_oss_dir + "/predictor_fp16.plan")
            local_path.append(
                os.path.join(self.root_dir, f"{model_card}/encoder_fp16.plan")
            )
            local_path.append(
                os.path.join(self.root_dir, f"{model_card}/predictor_fp16.plan")
            )

        for online_path, local_path in zip(online_path, local_path):
            self.download(online_path, local_path)

        if model_type == "pt":
            with Path(local_path[0]).open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            data["ckpt_path"] = local_path[1]

            with Path(local_path[0]).open("w", encoding="utf-8") as f:
                yaml.dump(data, f)

    def _cal_auth(
        self,
    ):
        assert EDP_AK is not None and EDP_SK is not None, "Environment variable EDP_AK and EDP_SK must be set"
        ak = EDP_AK
        sk = EDP_SK

        timestamp_seconds = time.time()
        timestamp_milliseconds = int(timestamp_seconds * 1e3)
        string_to_encrypt = sk + "," + str(timestamp_milliseconds)
        encoded_string = string_to_encrypt.encode("utf-8")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(encoded_string)
        encrypted_string = sha256_hash.hexdigest()
        header = (
            "Digest " + ak + ";" + str(timestamp_milliseconds) + ";" + encrypted_string
        )

        return header

    def _create_model(self, model_card_id, desc, step, model_type):
        url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model/create"
        payload = json.dumps(
            {
                "modelCardId": model_card_id,
                "description": desc,
                "step": step,
                "modelType": model_type,
            }
        )
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": self._cal_auth(),
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def _get_model_card_by_name(self, model_card_name):
        url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-card/query?nameList={model_card_name}&pageNum=1&pageSize=-1"
        payload = {}
        headers = {"accept": "*/*", "Authorization": self._cal_auth()}
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        # print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def _get_model(self, model_id):
        url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model/query?modelId={model_id}&pageNum=1&pageSize=-1"
        payload = {}
        headers = {"accept": "*/*", "Authorization": self._cal_auth()}
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def _get_model_by_model_card_id(self, model_card_id):
        url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model/query?modelCardId={model_card_id}&pageNum=1&pageSize=-1"
        payload = {}
        headers = {"accept": "*/*", "Authorization": self._cal_auth()}
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def _get_model_oss_dir(self, models_info, model_type):
        for model_info in models_info:
            if model_info["modelType"] == model_type:
                return model_info["ossDir"]

    def _get_model_id(self, models_info, model_type):
        for model_info in models_info:
            if model_info["modelType"] == model_type:
                return model_info["id"]

    def _create_model_export_record(
        self, compileServerSn, modelCardId, sourceModelId, targetModelType, description
    ):
        url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-export-record/create"
        payload = json.dumps(
            {
                "compileServerSn": compileServerSn,
                "modelCardId": modelCardId,
                "sourceModelId": sourceModelId,
                "targetModelType": targetModelType,
                "description": description,
            }
        )
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": self._cal_auth(),
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def notify_compile_server(
        self,
    ):
        compiler_server_sn = "1425023298546"
        models_info = self._get_model_by_model_card_id(self.model_card_id)["data"][
            "records"
        ]
        onnx_model_id = self._get_model_id(models_info, "onnx")
        print(models_info)
        print(onnx_model_id)
        self._create_model_export_record(
            compiler_server_sn,
            int(self.model_card_id),
            int(onnx_model_id),
            "plan_fp16",
            "",
        )

    def get_unfinished_model_export_record(self, compileServerSn="1425023298546"):
        url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-export-record/query?compileServerSn={compileServerSn}&statusList=not_started,in_progress&pageNum=1&pageSize=-1"
        payload = {}
        headers = {"accept": "*/*", "Authorization": self._cal_auth()}
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def _get_source_and_target_oss_dir(self, model_info):
        records = model_info["data"]["records"][0]
        model_card_name = records["modelCardName"]
        onnx_oss_dir = records["sourceModelOssDir"]
        plan_oss_dir = records["targetModelOssDir"]
        model_export_record_id = records["id"]

        return model_card_name, model_export_record_id, onnx_oss_dir, plan_oss_dir

    def get_onnx_model(self, model_card_name, onnx_oss_dir):
        encoder_onnx_key = onnx_oss_dir + "/encoder.onnx"
        predictor_onnx_key = onnx_oss_dir + "/predictor.onnx"
        os.makedirs(os.path.join(self.root_dir, model_card_name), exist_ok=True)
        encoder_onnx_file = os.path.join(
            self.root_dir, f"{model_card_name}/encoder.onnx"
        )
        predictor_onnx_file = os.path.join(
            self.root_dir, f"{model_card_name}/predictor.onnx"
        )
        self.download(encoder_onnx_key, encoder_onnx_file)
        self.download(predictor_onnx_key, predictor_onnx_file)

    def report_model_export_progress(self, modelExportRecordId, progress):
        url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-export-record/report-progress"
        payload = json.dumps(
            {
                "modelExportRecordId": modelExportRecordId,
                "progress": progress,
                "Authorization": self._cal_auth(),
            }
        )
        headers = {"accept": "*/*", "Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def report_model_export_status(self, modelExportRecordId, status):
        url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-export-record/report-status"
        payload = json.dumps(
            {"modelExportRecordId": modelExportRecordId, "status": status}
        )
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": self._cal_auth(),
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data
