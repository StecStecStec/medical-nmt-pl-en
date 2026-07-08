# ==========================================================================
# DEPRECATED — DO NOT USE.
#
# This was an earlier training script (no best-model selection, no baseline,
# no test evaluation, no result persistence). It has been superseded by
# finetuning_best_validation.py, which is the single source of truth for the
# experiment. Kept only for history.
# ==========================================================================
import sys

print(
    "datatesting.py is DEPRECATED. Use finetuning_best_validation.py instead.",
    file=sys.stderr,
)
sys.exit(1)

import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainingArguments
import nltk
import evaluate
import torch
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

df_training = pd.read_csv("data_training_short.csv", header=None, names=["polish", "english"])
df_training = df_training.fillna("")
df_training["polish"] = df_training["polish"].astype(str)
df_training["english"] = df_training["english"].astype(str)

dataset_training = Dataset.from_pandas(df_training)

df_validation = pd.read_csv("data_validation_short.csv", header=None, names=["polish", "english"])
df_validation = df_validation.fillna("")
df_validation["polish"] = df_validation["polish"].astype(str)
df_validation["english"] = df_validation["english"].astype(str)

dataset_validation = Dataset.from_pandas(df_validation)

model_name = "Helsinki-NLP/opus-mt-pl-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples["polish"]
    targets = examples["english"]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    return model_inputs

tokenized_dataset_training = dataset_training.map(preprocess_function, batched=True)
tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)

model = MarianMTModel.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-marian",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

nltk.download("punkt")
nltk.download("punkt_tab")
metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds_arr = np.array(predictions)
    if preds_arr.ndim == 3:  # logits
        preds_arr = np.argmax(preds_arr, axis=-1)

    labels_arr = np.array(labels)
    labels_arr = np.where(labels_arr != -100, labels_arr, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds_arr, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_arr, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]

    references = [[lab] for lab in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=references)

    return {"bleu": result["bleu"]}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_training,
    eval_dataset=tokenized_dataset_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
