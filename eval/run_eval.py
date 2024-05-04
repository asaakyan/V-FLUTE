import argparse
import os
os.environ["HF_HOME"] = "/mnt/swordfish-pool2/models/transformers_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from transformers import set_seed
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_mod import eval_model
import json
from tqdm import tqdm
import pandas as pd
# from compute_metrics import get_f1_and_bertscore
import re
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

def label_extraction_rules(output):

    label, expl = None, output

    # If possible to separate the label and the explanation
    label_pattern = re.compile(r'LABEL:|Label:|label:')
    match = label_pattern.search(output)
    if match:
        label_text = output[match.end():].strip()
        if "neither" in label_text.lower():
            label = None
        elif "contradict" in label_text.lower():
            label = 0
        elif "entail" in label_text.lower():
            label = 1
        expl = output[:match.start()].strip()
        # if expl is empty, for example if format is Label: x\nExpl
        if not expl:
            if label == 1:
                label_pattern = re.compile(r'entailment|Entailment|entails|Entails')
                match = label_pattern.search(output)
                if match: expl = output[match.end():].strip()
            elif label == 0:
                label_pattern = re.compile(r'contradiction|Contradiction|contradicts|Contradicts')
                match = label_pattern.search(output)
                if match: expl = output[match.end():].strip()
        # if still expl is empty, assign the output as expl
        if not expl:
            expl = output
    # otherwise, apply heuristics to get the label and assume explanation is same as full output
    else:
        x = output.lower()
        if ("not possible to definitively label" in x 
            or "neither" in x 
            or "does not support or contradict" in x
            or "entailment or contradiction" in x):
            label = None
        elif ((("entail" in x) )
            or
            (("supports the claim" in x))
            or 
            ("is consistent" in x)
            or
            ("image can be seen as indirectly supporting" in x)
            or
            ("appears to be consistent with the claim" in x)
            or
            ("in harmony with" in x)
            or
            ("is in agreement" in x)
            or
            (("confirms the claim" in x) )
            ):
            label = 1
        elif (("contradict" in x) 
            or ("appears to contest" in x)):
            label = 0
        else:
            label = None

    return label, expl

def main(args):
    os.environ["HF_HOME"] = args.transformers_cache
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    set_seed(args.seed)

    if args.ccot:
        print("Structured CoT enabled")
        with open ("ccot_prompt.txt", "r") as f:
            ccot_prompt = f.read()

    print('Loading model')
    # if not lora, essentially
    if args.model_base:
        model_path = f"{args.checkpoints_path}/{args.model_dir}"
        model_base = f"{args.transformers_cache}/{args.model_base}"
    else:
        model_path = f"{args.transformers_cache}/{args.model_dir}"
        model_base = None

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_4bit=False
    )
    print(f'Model {model_name} loaded')

    # loading data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # obtain preds
    res = []
    max_preds = args.max_preds
    for row in tqdm(data[:max_preds]):

        prompt = row['conversations'][0]['value'][8:]
        # image_file = f"{args.image_path}/{row['image']}"
        image_file = os.path.join(args.image_path, row['image'])
        # print(image_file)
        if args.ccot:
            # get prompt to generate scene graph
            sg_prompt = "Question: " + " ".join(prompt.split("?")[:-1]).replace("  ", " ").strip()
            sg_prompt += "\n" + ccot_prompt
            infer_args = type('Args', (), {
            "model_name": model_name,
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "query": sg_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": args.temp,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_len_sg
            })()
            scene_graph = eval_model(infer_args)
            # pprint(scene_graph)
            # get prompt to generate prediction
            prompt_ctx = f"Scene Graph:\n{scene_graph}\nUse the image and scene graph as context and answer the following question:"
            prompt_q = prompt_ctx + f"\nQuestion:\n{prompt}"

            infer_args = type('Args', (), {
            "model_name": model_name,
                "model": model,
                "tokenizer": tokenizer,
                "image_processor": image_processor,
                "query": prompt_q,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": args.temp,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens
            })()
            output = eval_model(infer_args)
            label_pred, expl = label_extraction_rules(output)
            res.append([row['id'], row['phenomenon'], row['source_dataset'], row['image'], 
                        label_pred, expl, output, row['conversations'][1]["value"], scene_graph])

        else:
            infer_args = type('Args', (), {
            "model_name": model_name,
                "model": model,
                "tokenizer": tokenizer,
                "image_processor": image_processor,
                "query": prompt,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": args.temp,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens
            })()
            output = eval_model(infer_args)
            label_pred, expl = label_extraction_rules(output)
            res.append([row['id'], row['phenomenon'], row['source_dataset'], row['image'], 
                        label_pred, expl, output, row['conversations'][1]["value"]])

    # save preds to df
    if args.ccot:
        df = pd.DataFrame(res, columns=['id', 'phenomenon', 'source_dataset',
                                         'image_path', 'label', 'explanation', 
                                         'output', 'reference_output', 'scene_graph']) 
    else:
        df = pd.DataFrame(res, columns=['id', 'phenomenon', 'source_dataset',
                                        'image_path', 'label', 'explanation', 
                                        'output', 'reference_output']) 

    output_dir = f"{args.output_dir}/{model_name}{args.custom_name}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/preds.csv", index=False)
    print(f"Saved to {output_dir}/preds.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation script")
    parser.add_argument("--transformers_cache", type=str, 
                        default="/mnt/swordfish-pool2/models/transformers_cache",
                        help="Path to the transformers cache")
    parser.add_argument("--cuda_visible_devices", type=str, 
                        default="0", help="CUDA visible devices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, 
                        default="../data/flute-v-llava/valid_data_v3.json",
                        help="Path to the data file")
    parser.add_argument("--image_path", type=str, 
                        default="/mnt/swordfish-pool2/asaakyan/visEntail/data/v-flute",
                        help="Path to the image directory")
    parser.add_argument("--checkpoints_path", type=str, 
                        default="/mnt/swordfish-pool2/asaakyan/visEntail/checkpoints",
                        help="Path to the checkpoints directory")
    parser.add_argument("--model_dir", type=str, 
                        default="",
                        help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="preds",
                        help="Path to the output directory")
    parser.add_argument("--model_base", type=str, 
                        default=None,
                        help="Path to the base model")
    parser.add_argument("--num_beams", type=int, 
                        default=3,
                        help="Number of beams")
    parser.add_argument("--temp", type=float, default=0,
                        help="Temperature")
    parser.add_argument("--custom_name", type=str, default="",
                        help="custom save name e.g. _temp0")
    parser.add_argument("--top_p", type=float, default=None,
                        help="top_p")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="max_new_tokens")
    parser.add_argument("--ccot", type=bool, default=False,
                        help="use compositional chain of thought")
    parser.add_argument("--max_len_sg", type=int, default=256,
                        help="max_len_sg")
    parser.add_argument("--max_preds", type=int, default=1000,
                        help="max_preds")

    args = parser.parse_args()
    main(args)
