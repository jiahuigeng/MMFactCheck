import os
from shutil import copyfile
from utils import *
from utils_llm import *
from prompts import *
import argparse
import logging

def parse_comma_separated_list(value):
    values = value.split(',')
    return [item.strip() for item in values]
def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataset = get_dataset(args.dataset)

    tgt_folder = os.path.join("results", "authen")
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    ori_file = os.path.join("dataset", args.dataset, args.dataset+".csv")
    res_file = os.path.join(tgt_folder, '_'.join([args.task, args.dataset, args.model, args.model_size]) + '.csv')
    if not os.path.exists(res_file):
        copyfile(ori_file, res_file)

    for mode in parse_comma_separated_list(args.mode):
        for i in range(args.repeat):
            col_name = '_'.join([args.dataset, args.model, args.model_size, mode, str(i)])
            if col_name not in dataset.columns:
                dataset[col_name] = None

    from lmdeploy import pipeline, TurbomindEngineConfig
    from lmdeploy.vl import load_image
    if args.model_size == "small":
        model_id = 'OpenGVLab/InternVL2-4B'
    elif args.model_size == "medium":
        model_id = 'OpenGVLab/InternVL2-8B'
    elif args.model_size == "large":
        model_id = 'OpenGVLab/InternVL2-26B'

    pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192))

    for mode in parse_comma_separated_list(args.mode):
        pmp_template = AUTHEN_PROMPTS[mode]
        for idx, row in dataset.iterrows():
            logger.info(f"current index {idx}")
            image_id, label = row["image_id"], row['label']

            # if mode == 'icl1':
            #     total_prompt = pmp_template.format(claim)
            # elif mode == 'icl2':
            #     total_prompt = pmp_template.format()
            # else:
            total_prompt = pmp_template

            for samp_idx in range(args.repeat):
                col_name = '_'.join([args.dataset, args.model, args.model_size, mode, str(samp_idx)])
                if pd.isna(dataset.iloc[idx][col_name]) and os.path.exists(image_id):
                    # pass
                    response = pipe((total_prompt, load_image(image_id))).text
                    logger.info(response)
                    dataset.at[idx, col_name] = response
                    dataset.to_csv(res_file, index=False)
                    logger.info(f"{col_name} is saved!")
                else:
                    logger.info(f"{col_name} is filled")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='authen')
    parser.add_argument('--dataset', type=str, default='real_and_fake_face')
    parser.add_argument('--model', type=str, default="intern")
    parser.add_argument('--model_size', type=str, default="medium")
    parser.add_argument('--mode', type=str, default="cot")
    parser.add_argument('--repeat', type=int, default=1)

    args = parser.parse_args()
    main(args)
