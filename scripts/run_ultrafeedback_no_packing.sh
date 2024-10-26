dataset="HuggingFaceH4/ultrafeedback_binarized" 
model="alignment-handbook/zephyr-7b-sft-full"
max_length=1024
max_prompt_length=512
batch_size=4
output_dir="ultrafeedback_no_packing"

accelerate launch --config_file 'configs/zero3.yaml'  train_dpo.py \
  --dataset_name=$dataset \
  --model_name_or_path=$model \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --beta 0.01 \
  --learning_rate 5e-7 \
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
  --dataset_train_split train_prefs \
  --dataset_test_split test_prefs \
  --keep_columns chosen rejected \
  --num_train_epochs 1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --attn_implementation flex_attention \
  --prefix_sharing
