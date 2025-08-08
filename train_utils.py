import torch
# Revert to using the original Seq2Seq trainer and arguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import set_seed
import bitsandbytes.optim as bnb_optim #i added this

def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, data_collator=None):
    set_seed(run)
    model = AutoModelForCausalLM.from_pretrained(
      args.from_pretrained,
      torch_dtype=torch.bfloat16,
      device_map="auto",
    )

    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'
    logging_dir = f'logs/{config_dir}/{run}'

    logging_strategy = 'steps' if not args.no_log else 'no'
    if args.no_log:
        logging_dir = None

    # Use Seq2SeqTrainingArguments which includes the predict_with_generate flag
    # Use Seq2SeqTrainingArguments which includes the predict_with_generate flag
    # Use Seq2SeqTrainingArguments which includes the predict_with_generate flag
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        eval_strategy='steps',
        eval_steps=200,                        # Evaluate every 200 steps
        save_strategy='steps',
        save_steps=200,                        # Save a checkpoint every 200 steps
        logging_dir=logging_dir,
        logging_strategy='steps',              # Log metrics every 200 steps
        logging_steps=200,
        max_steps=2625,                        # Total steps for 3 epochs
        learning_rate=args.lr,                 # Keep this as a tunable argument
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=32,        # Your specified batch size
        per_device_eval_batch_size=32,         # Use same batch size for evaluation
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=1,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
    )

    optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=args.lr)#i had to add this
    # Use the Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        optimizers=(optimizer, None),#I had to add this
    )
    
    trainer.train()
    print(f"finished training")
