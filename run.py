import argparse
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader, SafetyDatasetLoader
from metrics import compute_metrics_text_aux, compute_metrics_equation_aux, compute_metrics_text, compute_metrics_equation
from train_utils import train_and_evaluate

def run(args):
    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
    elif args.dataset == 'asdiv': # NOTE: for augmenting SVAMP only
        dataset_loader_svamp = SVAMPDatasetLoader()
        dataset_loader_asdiv = ASDivDatasetLoader()
    elif args.dataset == 'safety':
        dataset_loader = SafetyDatasetLoader()
    else:
        raise ValueError("Invalid dataset specified.")

    if args.dataset == 'asdiv':
        datasets_svamp = dataset_loader_svamp.load_from_json()
        datasets_asdiv = dataset_loader_asdiv.load_from_json()
        datasets = DatasetDict({
            'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
            'test': datasets_svamp['test']
        })
    else:
        datasets = dataset_loader.load_from_json()

    if args.llm:
        if args.llm == 'palm':
            if args.dataset == 'asdiv':
                train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
                train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
                train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
                train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
                test_llm_rationales, test_llm_labels = dataset_loader_svamp.load_llm_preds(split='test')
            else:
                train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
                test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
        elif args.llm == 'gpt':
            train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
        else:
            raise ValueError("Invalid LLM for rationale loading specified.")

        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
        
        if dataset_loader.has_valid:
            if args.llm == 'palm':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
            else: # gpt
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
            datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
            datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)

    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if not dataset_loader.has_valid:
        train_temp_split = datasets['train'].train_test_split(test_size=0.1, seed=args.run)
        valid_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=args.run)
        datasets = DatasetDict({
            'train': train_temp_split['train'],
            'valid': valid_test_split['train'],
            'test': valid_test_split['test']
        })

    if args.label_type == 'llm' and args.llm is not None:
        datasets['train'] = datasets['train'].rename_column('label', 'gt_label')
        datasets['train'] = datasets['train'].rename_column('llm_label', 'label')
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')
    elif args.label_type != 'gt':
        raise ValueError("Invalid label_type specified.")

    #### Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )

    if args.model_type == 'standard':
        def tokenize_function(examples):
            messages_list = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ]
                for prompt, label in zip(examples['input'], examples['label'])
            ]
            tokenized_inputs = tokenizer.apply_chat_template(
                messages_list,
                padding='max_length',
                max_length=args.max_input_length,
                truncation=True,
                add_generation_prompt=False,
            )
            return {
                "input_ids": tokenized_inputs,
                "labels": tokenized_inputs,
            }
    elif args.model_type == 'task_prefix':
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
            
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            return model_inputs
    else:
        raise ValueError("Invalid model_type specified.")

    remove_cols = ['input', 'label']
    if args.label_type == 'llm':
        remove_cols.extend(['gt_label', 'llm_label'])
    if 'rationale' in datasets['train'].column_names:
        remove_cols.append('rationale')
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=list(set(remove_cols) & set(datasets['train'].column_names))
    )

    if args.model_type == 'standard':
        compute_metrics = compute_metrics_text_aux(tokenizer) if args.dataset not in ['svamp', 'asdiv'] else compute_metrics_equation_aux(tokenizer)
    else: # task_prefix
        compute_metrics = compute_metrics_text(tokenizer) if args.dataset not in ['svamp', 'asdiv'] else compute_metrics_equation(tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics, data_collator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--model_type', type=str, default='standard')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default=None)
    
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer_name', type=str, default='AdamW8bit')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    parser.add_argument('--lora_r', type=int, default=16, help="LoRA attention dimension (rank).")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout probability.")

    parser.add_argument('--resume_from_adapter', type=str, default=None, help="Path to a saved LoRA adapter to continue training from.")
    
    args = parser.parse_args()
    run(args)

