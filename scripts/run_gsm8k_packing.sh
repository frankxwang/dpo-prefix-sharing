dataset="fxwang/easy-math-trees_v2-round-2" 
model="round-1-model"
max_length=16384
packing_length=16384
batch_size=1
output_dir="gsm8k_packing-llama-3p2-1B-v2"

accelerate launch --config_file 'configs/zero3.yaml' train_dpo.py \
  --dataset_name=$dataset \
  --model_name_or_path=$model \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --beta 0.3 \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --bf16 \
  --logging_first_step \
  --no_remove_unused_columns \
  --output_dir $output_dir \
  --max_length $max_length \
  --gradient_checkpointing True \
  --save_strategy no \
  --dataset_train_split train \
  --num_train_epochs 1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --attn_implementation flex_attention \
  --prefix_sharing \
  --enable_packing \
  --packing_length $((packing_length*batch_size))