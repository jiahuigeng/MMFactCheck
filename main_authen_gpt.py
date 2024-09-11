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
    res_file = os.path.join(tgt_folder, '_'.join([args.task, args.dataset, args.model]) + '.csv')

    if not os.path.exists(res_file):
        copyfile(ori_file, res_file)

    dataset = pd.read_csv(res_file)
    for mode in parse_comma_separated_list(args.mode):
        for i in range(args.repeat):
            col_name = '_'.join([args.dataset, args.model, mode, str(i)])
            if col_name not in dataset.columns:
                dataset[col_name] = None

    model = get_gpt_model()
    for mode in parse_comma_separated_list(args.mode):
        pmp_template = AUTHEN_PROMPTS[mode]
        for idx, row in dataset.iterrows():
            logger.info(f"current index {idx}")
            image_id, label = row["image_id"], row['label']
            origin_label = row['origin_label']
            # if origin_label != 'Fake':
            #     continue
            print(image_id)

            # if mode == 'icl1':
            #     total_prompt = pmp_template.format(claim)
            # elif mode == 'icl2':
            #     total_prompt = pmp_template.format()
            # else:
            #     total_prompt = pmp_template
            total_prompt = pmp_template
            for samp_idx in range(args.repeat):
                col_name = '_'.join([args.dataset, args.model, mode, str(samp_idx)])
                if pd.isna(dataset.iloc[idx][col_name]) and os.path.exists(image_id):
                    response = prompt_gpt4(model, total_prompt, image_id)
                    logger.info(response)
                    dataset.at[idx, col_name] = response
                    dataset.to_csv(res_file, index=False)
                    logger.info(f"{col_name} is saved!")
                else:
                    logger.info(f"{col_name} is filled")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='authen')
    parser.add_argument('--dataset', type=str, default='post4v_real_or_fake') # 'real_and_fake_face'
    parser.add_argument('--model', type=str, default="gpt4v")
    parser.add_argument('--mode', type=str, default="cot")
    parser.add_argument('--repeat', type=int, default=1)


    args = parser.parse_args()
    main(args)
