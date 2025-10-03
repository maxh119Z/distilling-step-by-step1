import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
from peft import get_peft_model, LoraConfig

def get_config_dir(args):
    """Generates a directory name for saving model artifacts."""
    lora_config_str = f"lora_r{args.lora_r}_alpha{args.lora_alpha}"
    return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{lora_config_str}'

def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, data_collator=None):
    """
    Loads a model, applies LoRA, and runs the training and evaluation loop
    using the standard Hugging Face Trainer.
    """
    set_seed(run)
    
    # 1. Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Set up Training Arguments
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=500,
        logging_first_step=True,
        disable_tqdm=False,
        logging_steps=10,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_steps,
        seed=run,
        bf16=args.bf16,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=True,  # Loads the best adapter at the end
        save_total_limit=2,
    )

    # 4. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=None,
        data_collator=data_collator,
    )

    # 5. Start Training
    print("Starting LoRA training...")
    trainer.train()
    print("Training finished.")

    # 6. Save the final LoRA adapter
    final_model_path = f"fine-tuned-models/{config_dir}/{run}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ Final LoRA adapter saved successfully to: {final_model_path}")

