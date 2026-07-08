"""Fine-tune Helsinki-NLP/opus-mt-pl-en (MarianMT) for medical PL->EN translation.

SINGLE SOURCE OF TRUTH for the training experiment. Core training logic and
hyperparameters are preserved exactly as originally run. This script only adds
instrumentation on top:
  - training_log.json : per-epoch mean train loss, eval loss, BLEU, chrF
  - results.json      : baseline vs fine-tuned test-set BLEU and chrF (+ delta)

(The dead nltk punkt downloads from the original were removed — they were never
used for scoring. Nothing about the optimizer, schedule, or data changed.)
"""

import torch
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from data_utils import load_translation_dataset, build_preprocess
from metrics_utils import build_compute_metrics, save_training_log, save_results

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model_name = "Helsinki-NLP/opus-mt-pl-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

dataset_training = load_translation_dataset("data_training_short.csv")
dataset_validation = load_translation_dataset("data_validation_short.csv")
dataset_testing = load_translation_dataset("data_testing_short.csv")

preprocess_function = build_preprocess(tokenizer, max_length=128)
tokenized_dataset_training = dataset_training.map(preprocess_function, batched=True)
tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)
tokenized_dataset_testing = dataset_testing.map(preprocess_function, batched=True)

model = MarianMTModel.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-marian",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_strategy="steps",
    logging_steps=100,
    predict_with_generate=True,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# BLEU + chrF (chrF added for medical-domain MT; BLEU stays the best-model metric).
compute_metrics = build_compute_metrics(tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_training,
    eval_dataset=tokenized_dataset_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Test performance BEFORE training (baseline).
baseline_trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset_testing,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

baseline_metrics = baseline_trainer.evaluate()
print("Baseline performance:", baseline_metrics)

trainer.train()
trainer.save_model("./finetuned-marian-best")

# Persist the per-epoch training curve (loss / BLEU / chrF).
save_training_log(trainer.state.log_history, "training_log.json")

finetuned_model = MarianMTModel.from_pretrained("./finetuned-marian-best")

finetuned_trainer = Seq2SeqTrainer(
    model=finetuned_model,
    args=training_args,
    eval_dataset=tokenized_dataset_testing,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

finetuned_metrics = finetuned_trainer.evaluate()
print("Fine-tuned performance:", finetuned_metrics)

# Permanently record baseline vs fine-tuned test scores and the delta.
save_results(
    baseline_metrics,
    finetuned_metrics,
    "results.json",
    model_info={
        "base_model": model_name,
        "task": "medical PL->EN translation",
        "test_set": "data_testing_short.csv",
    },
)
