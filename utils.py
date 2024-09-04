import time

import pandas as pd
import os
import json
from google.cloud import vision
from tqdm import tqdm

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


def detect_web(path, how_many_queries=50):
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




if __name__ == "__main__":
    pass
    # path = "dataset/mr2/test/image/0.jpg"
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
    convert_file("mr2")