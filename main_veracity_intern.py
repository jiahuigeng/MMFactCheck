from shutil import copyfile
import logging
from utils import *
from utils_llm import *
from prompts import *
import argparse
from lmdeploy.vl.constants import IMAGE_TOKEN


def parse_comma_separated_list(value):
    values = value.split(',')
    return [item.strip() for item in values]

def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # dataset = get_dataset(args.dataset)
    if not os.path.exists("results"):
        os.makedirs("results")

    ori_file = os.path.join("dataset", args.dataset, args.dataset+".csv")
    res_file = os.path.join("results", '_'.join([args.dataset, args.model, args.model_size, args.mode]) + '.csv')

    if not os.path.exists(res_file):
        copyfile(ori_file, res_file)

    dataset = pd.read_csv(res_file)

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
        pmp_template = VERACITY_PROMPTS[mode]

        for idx, row in dataset.iterrows():
            claim, image_id, label = row['claim'], row["image_id"], row['label']
            logger.info(f"current index {idx}")
            if args.debug == "True" and idx > 5:
                break

            if args.mode == "icl1":
                total_prompt = pmp_template.format(DEMONSTRATIONS[str(idx%8)]["claim"], DEMONSTRATIONS[str(idx%8)]["label"], DEMONSTRATIONS[str(idx%8)]["reasoning"], claim)
            elif args.mode == "icl2":
                total_prompt = pmp_template.format(DEMONSTRATIONS[str(idx%8)]["claim"], DEMONSTRATIONS[str(idx%8)]["label"], DEMONSTRATIONS[str(idx%8)]["reasoning"],
                                                   DEMONSTRATIONS[str((idx+1)%8)]["claim"], DEMONSTRATIONS[str((idx+1)%8)]["label"], DEMONSTRATIONS[str((idx+1)%8)]["reasoning"], claim, )
            else:
                total_prompt = pmp_template.format(claim)
            print(total_prompt)

            for samp_idx in range(args.repeat):
                col_name = '_'.join([args.dataset, args.model, args.model_size, mode, str(samp_idx)])
                if pd.isna(dataset.iloc[idx][col_name]):
                    if args.mode == "icl1":
                        response = pipe((total_prompt, [load_image(DEMONSTRATIONS[str(idx%8)]['image_id']), load_image(image_id)])).text
                    elif args.mode == "icl2":
                        response = pipe((total_prompt, [load_image(DEMONSTRATIONS[str(idx%8)]['image_id']), load_image(DEMONSTRATIONS[str((idx+1)%8)]['image_id']),
                                                        load_image(image_id)])).text
                    else:
                        response = pipe((total_prompt, load_image(image_id))).text

                    dataset.at[idx, col_name] = response
                    dataset.to_csv(res_file, index=False)
                    logger.info(f"{col_name} is saved")
                else:
                    logger.info(f"{col_name} is filled")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mr2')
    parser.add_argument('--model', type=str, default='intern')
    parser.add_argument('--mode', type=str, default='icl2')
    parser.add_argument('--model_size', type=str, default="small")
    parser.add_argument('--debug', type=str, default='True')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()

    main(args)



