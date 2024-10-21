from trl.commands.cli_utils import DPOScriptArguments, TrlParser

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trainer import DPOTrainer
from config import DPOConfig

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    model = model_config.model_name_or_path
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    if training_args.prefix_sharing:
        assert model_config.attn_implementation == "flex_attention", "Must set --attn_implementation=flex_attention for prefix sharing attention mask support"
    if model_config.attn_implementation == "flex_attention":
        # because of compilation, batch sizes need to match so we don't trigger a recompilation
        assert training_args.per_device_eval_batch_size == training_args.per_device_train_batch_size, "Must have equal train and eval batch sizes for FlexAttention support"
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = model_config.model_name_or_path
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]


    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)

    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split]
    else:
        eval_dataset = None

    if training_args.max_train_samples and training_args.max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(training_args.max_train_samples))
    
    if eval_dataset and training_args.max_eval_samples and training_args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(training_args.max_eval_samples))
    ################
    # Training
    ################

    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)