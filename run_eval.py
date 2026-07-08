"""Standalone evaluation: baseline vs fine-tuned model on the medical test set.

Computes BLEU and chrF for both the original Helsinki-NLP/opus-mt-pl-en model
and the fine-tuned model, then writes results.json. Does NOT require retraining
— run it directly after cloning the repo (with the model present):

    python run_eval.py
    python run_eval.py --test_csv data_testing_short.csv --out results.json

Named run_eval.py (not evaluate.py) on purpose: a module named evaluate.py
would shadow the `evaluate` library import.
"""

import argparse

import torch
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from data_utils import load_translation_dataset, build_preprocess
from metrics_utils import build_compute_metrics, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Helsinki-NLP/opus-mt-pl-en")
    parser.add_argument("--finetuned_dir", default="./finetuned-marian-best")
    parser.add_argument("--test_csv", default="data_testing_short.csv")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    tokenizer = MarianTokenizer.from_pretrained(args.base_model)
    preprocess = build_preprocess(tokenizer, max_length=128)
    test_ds = load_translation_dataset(args.test_csv).map(preprocess, batched=True)

    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval-tmp",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # portable: CPU clones run in fp32
        report_to="none",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    compute_metrics = build_compute_metrics(tokenizer)

    def evaluate_model(model_ref):
        model = MarianMTModel.from_pretrained(model_ref)
        data_collator.model = model
        trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        return trainer.evaluate()

    baseline_metrics = evaluate_model(args.base_model)
    print("Baseline performance:", baseline_metrics)

    finetuned_metrics = evaluate_model(args.finetuned_dir)
    print("Fine-tuned performance:", finetuned_metrics)

    results = save_results(
        baseline_metrics,
        finetuned_metrics,
        args.out,
        model_info={
            "base_model": args.base_model,
            "finetuned_dir": args.finetuned_dir,
            "task": "medical PL->EN translation",
            "test_set": args.test_csv,
        },
    )
    print(f"Wrote {args.out}. Delta:", results["delta"])


if __name__ == "__main__":
    main()
