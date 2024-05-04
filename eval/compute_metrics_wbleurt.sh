#!/bin/bash

# for MODEL in "gemini"; do
#     for FS in "zs" "fs5" "fs10"; do
#         python extract_label_and_expl.py \
#             --input_file preds/${MODEL}/${MODEL}_${FS}.csv \
#             --output_file preds/${MODEL}/normalized_${MODEL}_${FS}.csv
#         python compute_metrics_bscore_bleurt.py \
#             --pred preds/${MODEL}/normalized_${MODEL}_${FS}.csv \
#             --output_dir "f1_res_${FS}"
#     done
# done

# for MODEL in "gpt4" "claude" "gemini"; do
#     for FS in "zs" "fs5" "fs10"; do
#         python extract_label_and_expl.py \
#             --input_file preds/${MODEL}/${MODEL}_${FS}.csv \
#             --output_file preds/${MODEL}/normalized_${MODEL}_${FS}.csv
#         python compute_metrics_bscore_bleurt.py \
#             --pred preds/${MODEL}/normalized_${MODEL}_${FS}.csv \
#             --output_dir "f1_res_${FS}"
#     done
# done

# MODEL="llava-v1.5-7b-evil-vflute-v2-lora_checkpoint-4380"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.5-7b-evil-v2-lora_checkpoint-4310"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.5-7b-vflute-v2-lora_checkpoint-144"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.6-mistral-7b"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.6-mistral-7b_sg"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.6-34b"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv

# MODEL="llava-v1.6-34b_sg"
# python extract_label_and_expl.py \
#     --input_file preds/${MODEL}/preds.csv \
#     --output_file preds/${MODEL}/normalized_preds.csv
# python compute_metrics_bscore_bleurt.py \
#     --pred preds/${MODEL}/normalized_preds.csv