from shutil import copyfile
from utils import *
from utils_llm import *
from prompts import *
import argparse

def parse_comma_separated_list(value):
    values = value.split(',')
    return [item.strip() for item in values]

def main(args):

    # dataset = get_dataset(args.dataname)
    if not os.path.exists("results"):
        os.makedirs("results")


    img_evidence_file = os.path.join("dataset", "retrieval_results", f"{args.dataname}_img_evidence.json")
    txt_evidence_file = os.path.join("dataset", "retrieval_results", f"{args.dataname}_txt_evidence.json")

    img_evidence = json.load(open(img_evidence_file, encoding='utf-8'))
    txt_evidence = json.load(open(txt_evidence_file, encoding='utf-8'))

    ori_file = os.path.join("dataset", args.dataname, args.dataname+".csv")
    res_file = os.path.join("results", '_'.join([args.dataname, args.model, args.model_size]) + '.csv')

    if not os.path.exists(res_file):
        copyfile(ori_file, res_file)

    dataset = pd.read_csv(res_file)

    todos = ['txt', 'img', 'both']

    for n_evi in range(1, 3):
        for todo in todos:
            col_name = '_'.join([args.dataname, args.model, args.model_size, args.mode, todo, str(n_evi)])
            if col_name not in dataset.columns:
                dataset[col_name] = None

    model, processor = get_llava16_model(args.model_size)

    pmp_template = VERACITY_PROMPTS[args.mode]
    for todo in todos:
        for idx, row in dataset.iterrows():
            claim, image_id, label = row['claim'], row["image_id"], row['label']
            if args.debug == "True" and idx > 5:
                break

            for n_evi in range(1, 3):
                if todo.startswith('txt'):
                    evidence = txt_evidence[str(idx)]['evidence_ll'][:n_evi]
                    evidence = [item[2] for item in evidence]
                    evidence = [item for item in evidence if len(item) < 10000]

                elif todo.startswith('img'):
                    evidence = img_evidence[str(idx)]['evidence_ll'][:n_evi]
                    evidence = [item for item in evidence if len(item) < 10000]
                    evidence = [item[2] for item in evidence]

                elif todo.startswith('both'):
                    evidence_txt = txt_evidence[str(idx)]['evidence_ll'][:n_evi]
                    evidence_txt = [item[2] for item in evidence_txt]
                    evidence_txt = [item for item in evidence_txt if len(item) < 10000]

                    evidence_img = img_evidence[str(idx)]['evidence_ll'][:n_evi]
                    evidence_img = [item for item in evidence_img if len(item) < 10000]
                    evidence_img = [item[2] for item in evidence_img]

                    evidence = evidence_txt + evidence_img


                evi_prompt = ""
                for item in evidence:
                    evi_prompt += "\nDocument: \n" + item.strip() + "\n"
                total_prompt = pmp_template.format(evi_prompt, claim)
                print(total_prompt)

                response = prompt_llava16(model, processor, image_id, total_prompt)
                print(response)
                col_name = '_'.join([args.dataname, args.model, args.model_size, args.mode, todo, str(n_evi)])
                dataset.at[idx, col_name] = response
                dataset.to_csv(res_file, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='mr2')
    parser.add_argument('--model', type=str, default='llava')
    parser.add_argument('--mode', type=str, default='rag')
    parser.add_argument('--model_size', type=str, default="small")
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--nrepeat', type=int, default=3)
    args = parser.parse_args()

    main(args)



