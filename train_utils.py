import torch
from transformers import AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import set_seed
# --- KEY CHANGE: Import PEFT libraries for LoRA ---
from peft import get_peft_model, LoraConfig

def get_config_dir(args):
    # Update the directory name to include LoRA parameters for better tracking
    lora_config_str = f"lora_r{args.lora_r}_alpha{args.lora_alpha}"
    return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{lora_config_str}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.lr}'

def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, data_collator=None):
    set_seed(run)
    
    # Load the base model as before
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    # --- KEY CHANGE: Define LoRA configuration ---
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )

    # --- KEY CHANGE: Apply LoRA to the model ---
    model = get_peft_model(model, peft_config)
    # This will print the percentage of trainable parameters, which will be very small!
    model.print_trainable_parameters()

    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}' 
    logging_dir = f'logs/{config_dir}/{run}'
    
    total_generation_length = args.max_input_length + args.max_new_tokens
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_strategy='steps',
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_steps,
        predict_with_generate=True,
        seed=run,
        bf16=args.bf16,
        generation_max_length=total_generation_length,
        report_to="tensorboard",
    )

    # The Trainer is smart enough to handle the PEFT model automatically.
    # We no longer need the custom optimizer block.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("🚀 Starting LoRA training...")
    trainer.train()
    print("✅ Training finished.")

    final_model_path = f"fine-tuned-models/{config_dir}/{run}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"🎉 Final model saved successfully to: {final_model_path}")
