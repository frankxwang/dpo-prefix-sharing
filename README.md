# Prefix Sharing for DPO 

Implementation of prefix sharing training for DPO with Flex Attention. 

![image](./assets/prefix_sharing.png)

Contains code to reproduce our work "Accelerating Direct Preference Optimization with Prefix Sharing", NeurIPS-FITML Workshop, 2024. [Paper link]

## Get Started
Installation instructions: 
- Python 3.10+
- CUDA 12.3 or above
- PyTorch 2.5.0+ (As of Sept 15th 2024, PyTorch 2.5.0 is avaiable in the [test channel](https://dev-discuss.pytorch.org/t/pytorch-2-5-release-branch-cut-for-pytorch-core-is-completed/2452/1))
- `pip install -r requirements.txt`

## Launch training
Our trainer is based on the DPO Trainer from 🤗TRL, and thus you can run training as you would for a typical DPO run. We further enable prefix sharing and sequence packing with the flags `--prefix_sharing` and `--enable_packing`. Some example commands are provided below for the [Capybara dataset](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized). 
### Prefix sharing
To reproduce our results for prefix sharing, run: 

```
accelerate launch --config_file 'configs/zero3.yaml'  train_dpo.py  --dataset_name=argilla/distilabel-capybara-dpo-7k-binarized --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.1 --per_device_train_batch_size 4 --learning_rate 1e-6 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --warmup_steps 20 --bf16 --logging_first_step --no_remove_unused_columns --output_dir --max_prompt_length 2180 --max_length 2842 --gradient_checkpointing True --save_strategy no --dataset_test_split test --dataset_train_split train --num_train_epochs 1 --dataloader_num_workers 4 --attn_implementation flex_attention --prefix_sharing 
```
Optionally, add reporting flags `--report_to wandb --run_name <my_run>` for WandB, etc

### Prefix sharing with sequence packing

Run:
```
accelerate launch --config_file 'configs/zero3.yaml'  train_dpo.py  --dataset_name=argilla/distilabel-capybara-dpo-7k-binarized --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.1 --per_device_train_batch_size 1 --learning_rate 1e-6 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --warmup_steps 20 --bf16 --logging_first_step --no_remove_unused_columns --output_dir outputs --max_prompt_length 2180 --max_length 2842 --gradient_checkpointing True --save_strategy no --dataset_test_split test --dataset_train_split train --num_train_epochs 1 --dataloader_num_workers 4 --attn_implementation flex_attention --prefix_sharing --enable_packing --packing_length 15872
```

NOTE: We calculate the "packing length" based on statistics for prefix shared inputs. In the case of Capybara, we choose a packing length of 3968 (calculated as $1.1 \times 95^{th}$ percentile of sequence lengths) and multiply by desired batch size (4) to get 15872. 

To run both the above commands, you use do: `bash benchmark/run_capybara.sh`

