import argparse
import os
import json
from tqdm import tqdm
import pandas as pd
import re
from pprint import pprint

def normalize_SG_expl(x):
    
    if not "scene graph" in x.lower():
        return x

    x = x.replace("and the scene graph's description", "")
    x = x.replace("and the scene graph both support", "supports")
    x = x.replace("and the scene graph together support", "supports")
    x = x.replace("and the scene graph", "")
    x = x.replace("or the scene graph", "")
    x = x.replace("n the scene graph", "n the image")
    x = x.replace("the scene graph shows", "the image shows")
    x = x.replace("the scene graph describes", "the image shows")
    x = x.replace("the scene graph indicates", "the image displays")
    x = x.replace("the scene graph states", "the image displays")
    x = x.replace("the scene graph explicitly states", "the image displays")
    x = x.replace("scene graph", "image")

    x = x.replace("  ", " ")
    x = x.strip()

    for word in x.split():
        if not "emoji" in word:
            if "_" in word:
                x = x.replace(word, word.replace("_", " ").replace('"', ""))

    return x

def label_extraction_rules(output):

    if not pd.notna(output):
        return None, None
    
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

    
    # addl expl proc rules
    expl = expl.replace("Contradiction.", "").replace("Entailment.", "").strip()
    expl = expl.replace("Contradiction:", "").replace("Entailment:", "").strip()
    expl = expl.replace("Contradiction\n", "").replace("Entailment\n", "").strip()
    if "Therefore," in expl:
        expl = expl.rsplit("Therefore,", 1)[0].strip()
    return label, expl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    # print(args)
    df = pd.read_csv(args.input_file)
    print("No explanation: ", df[df['explanation'].isna()].shape)
    if "output" not in df.columns:
        df['output'] = df['explanation']
    df['label'], df['explanation'] = zip(*df['output'].apply(label_extraction_rules))
    if "sg" in args.input_file:
        print("Normalizing SG explanations")
        df['explanation'] = df['explanation'].apply(normalize_SG_expl)
    df.to_csv(args.output_file, index=False)
    print(f"Saved to {args.output_file}")
