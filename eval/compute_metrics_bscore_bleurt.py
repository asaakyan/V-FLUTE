from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
os.environ["HF_HOME"] = "/mnt/swordfish-pool2/models/transformers_cache"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from evaluate import load
from transformers import set_seed
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm

def compute_f1_scores(df, output_dir,
                    #   thresholds=[0, 0.1, 0.2,
                    #               0.3, 0.4, 0.45,
                    #               0.5, 0.55, 0.6, 
                    #               0.7, 0.8, 0.9, 1], 
                    thresholds = np.arange(0, 1.01, 0.01),
                        model_name="llava"):

    f1_scores = []
    for threshold in thresholds:
        # Adjust predictions: If expl_score <= threshold, flip label_pred to ensure it's counted as incorrect
        # For binary classification, assuming labels are 0 and 1
        df['adjusted_label_pred'] = np.where(df['bleurt_bertscore'] <= threshold, 1 - df['label_true'], df['label_pred'])

        # Compute F1 score with adjusted predictions
        f1 = f1_score(df['label_true'], df['adjusted_label_pred'], average="macro")
        
        f1_scores.append((model_name, threshold, f1))
    
    scores_df_main = pd.DataFrame(f1_scores, columns=['model_name', 'threshold', 'f1'])
    scores_df_main.to_csv(f"{output_dir}/f1_at_thresh.csv", index=False)

    # the same, per dataset:
    scores_df_main['dataset'] = 'overall'
    # scores_df_main = scores_df_main[['model_name', 'dataset', 'threshold', 'f1']]
    scores_df_main = scores_df_main[['model_name', 'dataset', 'threshold', 'f1']]
    byDataset_dfs = [scores_df_main]
    for dataset in df['dataset'].unique():

        if dataset == "irfl":
            # for irfl, we should compute performance separately for idiom and non-idiom
            dataset_df = df[df['dataset'] == dataset].copy()
            # compute for idiom
            idiom_df = dataset_df[dataset_df['phenomenon'] == "idiom"].copy()
            print(f"idiom instances: {idiom_df.shape[0]}")
            f1_scores = []
            for threshold in thresholds:
                idiom_df['adjusted_label_pred'] = np.where(idiom_df['bleurt_bertscore'] <= threshold, 
                                                            1 - idiom_df['label_true'], 
                                                            idiom_df['label_pred'])
                # Compute F1 score with adjusted predictions
                f1 = f1_score(idiom_df['label_true'], 
                            idiom_df['adjusted_label_pred'], 
                            average="macro")
                dataset = "irfl_idiom"
                f1_scores.append((model_name, dataset, threshold, f1))  
            scores_df = pd.DataFrame(f1_scores, columns=['model_name', 'dataset', 'threshold', 'f1'])
            scores_df.to_csv(f"{output_dir}/irfl_idiom_f1_at_thresh.csv", index=False)
            byDataset_dfs.append(scores_df)
            # compute for metaphor and simile
            metaSimile_df = dataset_df[(dataset_df['phenomenon'] == "metaphor") 
                                       | (dataset_df['phenomenon'] == "simile")].copy()
            print(f"metahpor+simile instances: {metaSimile_df.shape[0]}")
            f1_scores = []
            for threshold in thresholds:
                metaSimile_df['adjusted_label_pred'] = np.where(metaSimile_df['bleurt_bertscore'] <= threshold, 
                                                            1 - metaSimile_df['label_true'], 
                                                            metaSimile_df['label_pred'])
                # Compute F1 score with adjusted predictions
                f1 = f1_score(metaSimile_df['label_true'], 
                            metaSimile_df['adjusted_label_pred'], 
                            average="macro")
                dataset = "irfl_metaphor_simile"
                f1_scores.append((model_name, dataset, threshold, f1))
            scores_df = pd.DataFrame(f1_scores, columns=['model_name', 'dataset', 'threshold', 'f1'])
            scores_df.to_csv(f"{output_dir}/irfl_metaphor_simile_f1_at_thresh.csv", index=False)
            byDataset_dfs.append(scores_df)
        else:
            dataset_df = df[df['dataset'] == dataset].copy()
            print(f"{dataset} instances: {dataset_df.shape[0]}")
            f1_scores = []
            for threshold in thresholds:
                dataset_df['adjusted_label_pred'] = np.where(dataset_df['bleurt_bertscore'] <= threshold, 
                                                            1 - dataset_df['label_true'], 
                                                            dataset_df['label_pred'])
                # Compute F1 score with adjusted predictions
                f1 = f1_score(dataset_df['label_true'], 
                            dataset_df['adjusted_label_pred'], 
                            average="macro")
                f1_scores.append((model_name, dataset, threshold, f1))  
            scores_df = pd.DataFrame(f1_scores, columns=['model_name', 'dataset', 'threshold', 'f1'])
            scores_df.to_csv(f"{output_dir}/{dataset}_f1_at_thresh.csv", index=False)
            byDataset_dfs.append(scores_df)
    
    pd.concat(byDataset_dfs).reset_index(drop=True).to_csv(f"{output_dir}/byDataset_f1_at_thresh.csv", index=False)
    return scores_df_main, byDataset_dfs

def get_f1_and_bertscore(pred, true, model_name, output_dir, dummy_bleurt=False):

    set_seed(42)
    eval_df = pred.merge(true, on='id', how='left', suffixes=('_pred', '_true'))
    # fill NA preds with incorrect label
    eval_df['label_pred'] = eval_df['label_pred'].fillna(1 - eval_df['label_true']).astype(int)
    # fill expl NA preds with empty string
    eval_df['explanation_pred'] = eval_df['explanation_pred'].fillna('')

    if dummy_bleurt:
        eval_df['bertscore'] = 1
        eval_df['bleurt'] = 1
    else:
        bertscore = load("bertscore")
        print("bertscore model loaded")
        results = bertscore.compute(predictions=eval_df['explanation_pred'], 
                                    references=eval_df['explanation_true'],
                                    model_type="microsoft/deberta-xlarge-mnli", 
                                    lang="en")
        eval_df['bertscore'] = results['f1']
        # eval_df['bertscore'] = 1
        bscore_mean = eval_df['bertscore'].mean()
        print('BERTScore:', bscore_mean)

        # flush cache
        torch.cuda.empty_cache()

        config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
        model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
        tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
        print("BLEURT model loaded")
        # device = "cuda:0,1,2,3"
        # model.to(device)
        model.eval()
        with torch.no_grad():
            bleurt_scores = []
            for i, row in tqdm(eval_df.iterrows()):
                inputs = tokenizer(row['explanation_true'], row['explanation_pred'], 
                            padding='longest', truncation=True, return_tensors='pt')
                # res = model(**inputs).logits.flatten().tolist()
                res = model(**inputs).logits.flatten().tolist()[0]
                bleurt_scores.append(res)
        eval_df['bleurt'] = bleurt_scores

        bleurt_mean = eval_df['bleurt'].mean()
        print('BLEURT:', bleurt_mean)

    eval_df['bleurt_bertscore'] = (eval_df['bertscore'] + eval_df['bleurt']) / 2
    eval_df.to_csv(f"{output_dir}/f1_bscore_bleurt.csv", index=False)
    f1_df, byDatasetF1_dfs = compute_f1_scores(eval_df, output_dir, model_name=model_name)
    print('f1@0:', f1_df[f1_df['threshold'] == 0]['f1'].values[0])
    print('f1@50:', f1_df[f1_df['threshold'] == 0.5]['f1'].values[0])
    print('f1@60:', f1_df[f1_df['threshold'] == 0.6]['f1'].values[0])
    print('f1@70:', f1_df[f1_df['threshold'] == 0.7]['f1'].values[0])
    print('f1@80:', f1_df[f1_df['threshold'] == 0.8]['f1'].values[0])
    print('f1@90:', f1_df[f1_df['threshold'] == 0.9]['f1'].values[0])
    for dataset in byDatasetF1_dfs:
        print(f"Dataset: {dataset['dataset'].values[0]}")
        print('f1@0:', dataset[dataset['threshold'] == 0]['f1'].values[0])
        print('f1@50:', dataset[dataset['threshold'] == 0.5]['f1'].values[0])
        print('f1@60:', dataset[dataset['threshold'] == 0.6]['f1'].values[0])
        print('f1@70:', dataset[dataset['threshold'] == 0.7]['f1'].values[0])
        print('f1@80:', dataset[dataset['threshold'] == 0.8]['f1'].values[0])
        print('f1@90:', dataset[dataset['threshold'] == 0.9]['f1'].values[0])

    return f1_df

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, 
                        help='Path to the predictions file')
    parser.add_argument('--true', type=str, 
                        # default="../data/v-flute_test_noimage.csv",
                        # default="../data/vflute-test_wContra.csv",
                        #  default="../data/vflute-test_wContra_fixed.csv",
                        default="../data/flute-v-llava-clean/vflute-v2-test.csv",
                        help='Path to the ground truth file')
    parser.add_argument('--dummy_bleurt', action='store_true',
                        help='Use dummy BLEURT and bertscore model')
    parser.add_argument('--freq_words_file', type=str, default=None,
                        help='Path to the frequent words file')
    parser.add_argument('--output_dir', type=str, default="f1_res",
                        help='Path to the output directory')
    args = parser.parse_args()

    model_name = args.pred.split('/')[1]
    if ".csv" in model_name:
        model_name = model_name.split("_")[0]
        
    output_dir = f"preds/{model_name}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    print("Output dir:", output_dir)
    print(f"Evaluating {model_name}")
    pred = pd.read_csv(args.pred)[['id', 'label', 'explanation']]
    true = pd.read_csv(args.true)
    true['label'] = true['label'].apply(lambda x: 1 if x == 'entailment' else 0)
    true['dataset'] = true['source_dataset']
    true.drop(columns=['transformed'], inplace=True)
    assert len(pred) == len(true)

    if args.freq_words_file:
        print("Removing frequent words")
        freq_words = pd.read_csv(args.freq_words_file)
        freq_words_set = set(freq_words['word'])
        pred['explanation'] = pred['explanation'].apply(lambda x: 
                            ' '.join([word for word in x.split() if word.lower() not in freq_words_set]))
        true['explanation'] = true['explanation'].apply(lambda x: 
                            ' '.join([word for word in x.split() if word.lower() not in freq_words_set]))

    get_f1_and_bertscore(pred, true, model_name, output_dir, args.dummy_bleurt)