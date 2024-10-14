
dataset="argilla/distilabel-capybara-dpo-7k-binarized" 
max_length=2842
max_prompt_length=2180
packing_length=3968
batch_size=4
accelerate launch --config_file 'configs/zero3.yaml'  train_dpo.py  --dataset_name=argilla/distilabel-capybara-dpo-7k-binarized --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.1 --per_device_train_batch_size $batch_size --learning_rate 1e-6 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --warmup_steps 20 --bf16 --logging_first_step --no_remove_unused_columns --output_dir outputs --max_prompt_length $max_prompt_length --max_length $max_length --gradient_checkpointing True --save_strategy no --dataset_test_split test --dataset_train_split train --num_train_epochs 1 --dataloader_num_workers 4 --attn_implementation flex_attention --prefix_sharing 

accelerate launch --config_file 'configs/zero3.yaml'  train_dpo.py  --dataset_name=argilla/distilabel-capybara-dpo-7k-binarized --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.1 --per_device_train_batch_size 1 --learning_rate 1e-6 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --warmup_steps 20 --bf16 --logging_first_step --no_remove_unused_columns --output_dir outputs --max_prompt_length $max_prompt_length --max_length $max_length --gradient_checkpointing True --save_strategy no --dataset_test_split test --dataset_train_split train --num_train_epochs 1 --dataloader_num_workers 4 --attn_implementation flex_attention --prefix_sharing --enable_packing --packing_length $((packing_length*batch_size))