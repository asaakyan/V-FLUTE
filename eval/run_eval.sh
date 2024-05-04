#!/bin/bash

7B-eViL-VFLUTE
python run_eval.py \
    --model_dir "llava-v1.5-7b-evil-vflue-v2-lora/checkpoint-4380" \
    --model_base "llava-v1.5-7b" \
    --num_beams 3 \
    --temp 0 \
    --max_new_tokens 256 \
    --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
    --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
    --cuda_visible_devices "0,1,2,3" \
    --seed 42

# #7B-FT
# python run_eval.py \
#     --model_dir "llava-v1.5-7b-vflute-v2-lora/checkpoint-144" \
#     --model_base "llava-v1.5-7b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --seed 42

# #7B
# python run_eval.py \
#     --model_dir "/llava-v1.6-mistral-7b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --seed 42

# #7B SG
# python run_eval.py \
#     --model_dir "/llava-v1.6-mistral-7b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --ccot true \
#     --max_len 256 \
#     --custom_name "_sg" \
#     --seed 42

# #34B
# python run_eval.py \
#     --model_dir "/llava-v1.6-34b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --seed 42

# #34B SG
# python run_eval.py \
#     --model_dir "/llava-v1.6-34b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --ccot true \
#     --max_len_sg 256 \
#     --custom_name "_sg" \
#     --seed 42

# #7B-eVil-v2
# python run_eval.py \
#     --model_dir "llava-v1.5-7b-evil-v2-lora/checkpoint-4310" \
#     --model_base "llava-v1.5-7b" \
#     --num_beams 3 \
#     --temp 0 \
#     --max_new_tokens 256 \
#     --data_path "../data/flute-v-llava-clean/vflute-v2-test.json" \
#     --image_path "/mnt/swordfish-pool2/asaakyan/visEntail/data/VFLUTE-v2" \
#     --cuda_visible_devices "0,1,2,3" \
#     --seed 42