from shutil import copyfile
from utils import *
from utils_llm import *
from prompts import *
import argparse

def parse_comma_separated_list(value):
    values = value.split(',')
    return [item.strip() for item in values]

def main(args):

    # dataset = get_dataset(args.dataset)
    if not os.path.exists("results"):
        os.makedirs("results")

    ori_file = os.path.join("dataset", args.dataset, args.dataset+".csv")
    res_file = os.path.join("results", '_'.join([args.dataset, args.model, args.mode]) + '.csv')

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
        pmp_template = VERACITY_PROMPTS[mode]
        for idx, row in dataset.iterrows():
            print(idx)
            claim, image_id, label = row['claim'], row["image_id"], row['label']
            if args.debug == 'True' and idx > 5:
                break

            total_prompt = pmp_template.format(claim)
            # print(total_prompt)

            for samp_idx in range(args.repeat):
                col_name = '_'.join([args.dataset, args.model, mode, str(samp_idx)])
                if pd.isnull(dataset.iloc[idx][col_name]):
                # response = prompt_llava16(model, processor, load_image_llava(image_id), total_prompt)
                    response = prompt_gpt4(model=model, prompt=total_prompt,image_id=image_id)
                    print(response)

                    dataset.at[idx, col_name] = response
                    dataset.to_csv(res_file, index=False)
                    print(f"{col_name} is saved")
                else:
                    print(f"{col_name} is filled")
                #

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='post4v')
    parser.add_argument('--model', type=str, default='gpt4v')
    parser.add_argument('--mode', type=str, default='cot')
    # parser.add_argument('--model_size', type=str, default="small")
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    main(args)



