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
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True, # Important for evaluating generative models
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
