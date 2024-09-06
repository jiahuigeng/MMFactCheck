import time

import pandas as pd
import os
import json
from google.cloud import vision
from tqdm import tqdm

from scrape_utils import *
import argparse
from google.oauth2 import service_account

credential_path = "pristine-valve-433607-g7-e816cdfde7d8.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def check_contains(text, ll):
    for item in ll:
        if item.lower() in text.lower():
            return True
    return False

def save_result(output,json_file_path):
    '''
    Save output results to a JSON file.
    '''
    try:
        if type(output)==str:
            user_data = json.loads(output)
            append_to_json(json_file_path, user_data)
        else:
            append_to_json(json_file_path, output)
    except json.JSONDecodeError:
        #The output was not well formatted
        pass

def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def concatenate_entry(d):
    '''
    For all keys in a dictionary, if a value is a list, concatenate it.
    '''
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = ';'.join(map(str, value))  # Convert list to a string separated by ';'
    return d

def append_to_json(file_path, data):
    '''
    Append a dict or a list of dicts to a JSON file.
    '''
    try:
        if not os.path.exists(file_path):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(file_path, 'w') as file:
                json.dump([], file)
        #Open the existing file
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            if type(data)==list:
                for d in data:
                    if type(d)==dict:
                        file_data.append(concatenate_entry(d))
            else:
                file_data.append(concatenate_entry(data))
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")

# def get_dataset(dataname):
    # if dataname == 'mr2':
    #     folder = './data/mr2/'
    #     csv_file = "dataset/mr2/mr2.csv"
    #
    #
    #     df_input = pd.read_csv(csv_file)
    #     # claims = df_input["claim"]
    #     df_input["image_id"] = "dataset/mr2/" +df_input["image_id"]
    #     df_input['label'] = df_input['label'].map({0: False, 1: True})
    #     return df_input
    #
    # elif dataname == 'fauxtography':
    #     folder = ".."
    #     csv_file = "dataset/fauxtography/fauxtography.csv"
    #     df_input = pd.read_csv(csv_file)
    #     # claims = df_input["claim_en"]
    #     # images = "dataset/mr2" +df_input["image_id"]
    #     # labels = df_input["veracity"]
    #     # df_input["claim"] = df_input["claim_en"]
    #     df_input["label"] = df_input["veracity"]
    #     return df_input


    # elif dataname == "post4v":
    #     folder = "../datasets/post4v"
    #     csv_file = "post4v.csv"

def get_dataset(dataname):
    file = os.path.join("dataset", f"{dataname}", f"{dataname}.csv")
    df_input = pd.read_csv(file)
    return df_input


def convert_file(dataname):
    if dataname == 'mr2':
        folder = './data/mr2/'
        csv_file = "dataset/mr2/mr2.csv"


        df_input = pd.read_csv(csv_file)
        # claims = df_input["claim"]
        # df_input["image_id"] = "dataset/mr2/" +df_input["image_id"]
        df_input['label'] = ~df_input['label']
        df_input.to_csv(csv_file, index=False)
        return df_input

    elif dataname == 'fauxtography':
        folder = ".."
        csv_file = "dataset/fauxtography/fauxtography.csv"
        df_input = pd.read_csv(csv_file)
        # claims = df_input["claim_en"]
        # images = "dataset/mr2" +df_input["image_id"]
        # labels = df_input["veracity"]
        # df_input["claim"] = df_input["claim_en"]
        df_input["label"] = df_input["veracity"]
        df_input.to_csv(csv_file, index=False)
        return df_input

def detect_ris_web(path, how_many_queries=50):
    """
    Detects web annotations given an image.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection

    page_urls = []
    matching_image_urls = {}

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            page_urls.append(page.url)
            if page.full_matching_images:
                matching_image_urls[page.url] = [image.url for image in page.full_matching_images]
            if page.partial_matching_images:
                matching_image_urls[page.url] = [image.url for image in page.partial_matching_images]
    else:
        print('No matching images found for ' + path)
        # print(annotations)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return page_urls, matching_image_urls

from duckduckgo_search import DDGS
def detect_duck_web(query, how_many_queries=30):
    """
    Detects web annotations given an image.
    """

    results = DDGS().text(query, max_results=how_many_queries)
    page_urls = []
    for item in results:
        page_urls.append(item['href'])
    print(f"page_urls: {page_urls}")
    time.sleep(5)
    return page_urls

def check_valid_url(url):
    valid = is_likely_html(url) and (not is_banned(url)) and (not is_obfuscated_or_encoded(url)) and (not is_fc_organization(url))
    return valid
def collect_txt_evidence(dataname, max_results=7, max_retrieval=30):
    '''
    collect evidence for claims in dataset.
    :param dataname:
    :return:
    total_evidence dict with
    idx, image_id, claim, evidence_ll
    '''
    dataset = get_dataset(dataname)

    evidence_file = os.path.join('dataset', 'retrieval_results', f'{dataname}_txt_evidence.json')
    total_evidence = {}
    if os.path.isfile(evidence_file):
       total_evidence = json.load(open(evidence_file, encoding='utf-8'))

    for idx, row in dataset.iterrows():
        print(idx)
        if str(idx) in total_evidence:
            print(f"{idx} is already")
            continue
        claim, image_id = row["claim"], row["image_id"]
        evidence_ll = []
        urls = detect_duck_web(claim, max_retrieval)
        cnt_valid_evi = 0
        for url in urls:
            if cnt_valid_evi >= max_results:
                break
            if check_valid_url(url):
                res = extract_info_trafilatura(url)
                if isinstance(res, dict):
                    cnt_valid_evi += 1
                    print(f"res {res['date']}")
                    evidence_ll.append((url, res['date'], res['text']))

        total_evidence[str(idx)] = {
            'claim': claim,
            'image_id': image_id,
            'evidence_ll': evidence_ll
        }
        json.dump(total_evidence, open(evidence_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

def collect_img_evidence(dataname, max_results=7, max_retrieval=30):
    img_evidence_file = os.path.join("dataset", 'retrieval_results', f'{dataname}_img_evidence.json')

    total_evidence = {}
    if os.path.exists(img_evidence_file):
        total_evidence = json.load(open(img_evidence_file, encoding='utf-8'))

    dataset = get_dataset(dataname)
    for idx, row in dataset.iterrows():
        print(idx)
        if str(idx) in total_evidence:
            print(f"{idx} is already")
            continue
        claim, image_id = row["claim"], row["image_id"]
        evidence_ll = []
        # urls = detect_duck_web(claim, max_retrieval)
        try:
            urls, _ = detect_ris_web(image_id, max_retrieval)
        except:
            continue
        cnt_valid_evi = 0
        for url in urls:
            if cnt_valid_evi >= max_results:
                break
            if check_valid_url(url):
                res = extract_info_trafilatura(url)
                if isinstance(res, dict):
                    cnt_valid_evi += 1
                    print(f"res {res['date']}")
                    evidence_ll.append((url, res['date'], res['text']))

        total_evidence[str(idx)] = {
            'claim': claim,
            'image_id': image_id,
            'evidence_ll': evidence_ll
        }
        json.dump(total_evidence, open(img_evidence_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)



def get_img_evidence(dataname):

    img_evidence_file = os.path.join("dataset", 'retrieval_results', f'{dataname}_img_evidence.json')

    total_evidence = {}
    if os.path.exists(img_evidence_file):
        total_evidence = json.load(open(img_evidence_file, encoding='utf-8'))

    evidence_imageid = dict()
    evidence_img_json = json.load(open(os.path.join("dataset", "retrieval_results", f"evidence_{dataname}_img.json")))
    for data in evidence_img_json:
        # print(data["image path"])
        if data["image path"] not in evidence_imageid:
            evidence_imageid[data["image path"]] = []
        evidence_imageid[data['image path']].append((data['evidence url'], data['date'], data['text']))

    dataset = get_dataset(dataname)
    cnt_exist, cnt_not_exist = 0, 0
    for idx, row in dataset.iterrows():
        if row["image_id"] in evidence_imageid and len(evidence_imageid[row["image_id"]])>=7:
            total_evidence[str(idx)] = {
                "claim": row["claim"],
                "image_id": row["image_id"],
                "evidence": evidence_imageid[row["image_id"]][:7]
            }
            print(f"found!")
            cnt_exist += 1
    print(cnt_exist)
    json.dump(total_evidence, open(img_evidence_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


    # img_evi_url = os.path.join("dataset", "retrieval_results", f"link_{dataset}_img.txt")
    # img_evi_url = open()
    # txt_evi_url = open(f"")



    # evidence_txt_file = os.path.join("dataset", "retrieval_results", f"evidence_{dataset}_txt.json")
    # evidence = None
    # return evidence


if __name__ == "__main__":
    # collect_img_evidence("fauxtography")

    collect_txt_evidence("fauxtography")
    # makeup_txt_evidence("mr2")
    # collect_txt_evidence("fauxtography")
    # parser = argparse.ArgumentParser()
    # get_img_evidence("mr2")
    # parser.add_argument('--dataname', type=str, default="")

    # get_evidence("fauxtography")
    # path = "dataset/mr2/test/ima
    # ge/0.jpg"
    # print(detect_web(path))
    # data = get_dataset("fauxto")
    # for idx, row in data.iterrows():
    #     # if idx > 4:
    #     #     break
    #     print(row["claim"], row["image_id"], row["label"])
    #     # print(os.path.exists(row["image_id"]))
    #     # if not os.path.exists(row["image_id"]):
    #     #     print(row["image_id"])

    # convert_file("fauxtography")
    # convert_file("mr2")