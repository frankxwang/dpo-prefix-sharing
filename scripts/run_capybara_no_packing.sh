dataset="argilla/distilabel-capybara-dpo-7k-binarized" 
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_length=2842
max_prompt_length=2180
batch_size=2
output_dir="capybara_no_packing"

accelerate launch --config_file 'configs/zero3.yaml' train_dpo.py \
  --dataset_name=$dataset \
  --model_name_or_path=$model \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --beta 0.1 \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --bf16 \
  --logging_first_step \
  --no_remove_unused_columns \
  --output_dir $output_dir \
  --max_prompt_length $max_prompt_length \
  --max_length $max_length \
  --gradient_checkpointing True \
  --save_strategy no \
  --dataset_train_split train \
  --num_train_epochs 1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --attn_implementation flex_attention \
  --prefix_sharing
