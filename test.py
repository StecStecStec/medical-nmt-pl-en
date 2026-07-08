# NOTE: Superseded by inference.py (adds CLI args + reusable translate()).
# Kept as a minimal scratch example; use `python inference.py` instead.
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainingArguments
import torch

model_name_path = "./finetuned-marian-best"
tokenizer = MarianTokenizer.from_pretrained(model_name_path)
model = MarianMTModel.from_pretrained(model_name_path)

sentence = "W przypadku dożylnego podania preparatu mogą występować niepożądane reakcje np. podwyższona ciepłota ciała, luźne stolce, brak łaknienia obniżone pragnienie oraz zapaść."

inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    translated_tokens = model.generate(**inputs, max_length=128)

translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("Translation:", translation)