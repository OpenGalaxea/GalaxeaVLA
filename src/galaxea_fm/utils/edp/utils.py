import os 
import json
import hashlib
import time 
import requests

from galaxea_fm.utils.edp.user import EDP_AK, EDP_SK


def cal_auth():
    timestamp_milliseconds = int(time.time() * 1000)
    string_to_encrypt = f"{EDP_SK},{timestamp_milliseconds}"
    encrypted_string = hashlib.sha256(string_to_encrypt.encode('utf-8')).hexdigest()
    return f"Digest {EDP_AK};{timestamp_milliseconds};{encrypted_string}"


def create_model_card(modelWarehouse, name, description, trainedBy, trainingTime, gitBranch, gitCommit, trainingDataSetIds):
    url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-card/create"
    payload = json.dumps({
        "modelWarehouse": modelWarehouse,
        "name": name,
        "description": description,
        "trainedBy": trainedBy,
        "trainingTime": trainingTime,
        "gitBranch": gitBranch,
        "gitCommit": gitCommit,
        "trainingDataSetIds": trainingDataSetIds
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth()
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()
  #  print(response.text)
# create_model_card("project-z", "task001", "this is desc", "wangruolin", "2025-02-09 10:00:00", "dev-branch", "adsfersdfs776", [11,22])


#########################
# ， model id
#########################
def create_model(model_card_id, desc, step, model_type):
    url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model/create"
    payload = json.dumps({
        "modelCardId": model_card_id,
        "description": desc,
        "step": step,
        "modelType": model_type
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth()
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()
# create_model(1, "this is desc", 1000, "pt")


#########################
#  model
#########################
def get_model(model_id):
    url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model/query?modelId={model_id}&pageNum=1&pageSize=-1"
    payload = {}
    headers = {
        'accept': '*/*',
        'Authorization': cal_auth()
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()


#########################
# 
#########################
def patch_label_for_raw_data(label_id, raw_data_id):
    patch_label_for_entity(label_id,  "raw_data", raw_data_id)


def patch_label_for_model_card(label_id, model_card_id):
    patch_label_for_entity(label_id,  "model_card", model_card_id)


def patch_label_for_entity(label_id, entity_type, entity_id):
    url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/label-mapping/create"
    payload = json.dumps({
        "labelId": label_id,
        "entityType": entity_type,
        "entityId": entity_id
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth()
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


#########################
# 
#########################
def delete_label_for_raw_data(label_id, raw_data_id):
    delete_label_for_entity(label_id,  "raw_data", raw_data_id)


def delete_label_for_model_card(label_id, model_card_id):
    delete_label_for_entity(label_id,  "model_card", model_card_id)


def delete_label_for_entity(label_id, entity_type, entity_id):
    url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/label-mapping/delete"
    payload = json.dumps({
        "labelId": label_id,
        "entityType": entity_type,
        "entityId": entity_id
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth()
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


#########################
# name
#########################
def get_model_card_by_name(model_card_name):
    url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/model-card/query?nameList={model_card_name}&pageNum=1&pageSize=-1"
    payload = {}
    headers = {
        'accept': '*/*',
        'Authorization': cal_auth()
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    data = json.loads(response.text)
    return data
# get_model_card_by_name("0720_task002-box1_anqi_rgb-norm")


#########################
# meta，：
#########################
def get_training_record_meta(training_record_name):
    url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/training-record/get-meta?name={training_record_name}"
    payload = {}
    headers = {
        'accept': '*/*',
        'Authorization': cal_auth()
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    data = json.loads(response.text)
    return data
# get_training_record_meta("test-001")
