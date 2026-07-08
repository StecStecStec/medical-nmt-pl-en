"""Translate a Polish medical sentence with the fine-tuned MarianMT model.

Quick sanity check for anyone cloning the repo:

    python inference.py
    python inference.py --sentence "Pacjent zgłasza bóle głowy."
"""

import argparse

import torch
from transformers import MarianTokenizer, MarianMTModel

# Sample drawn from the medical domain of the training data.
DEFAULT_SENTENCE = (
    "W przypadku dożylnego podania preparatu mogą występować niepożądane reakcje "
    "np. podwyższona ciepłota ciała, luźne stolce, brak łaknienia obniżone "
    "pragnienie oraz zapaść."
)


def translate(sentence, model_dir="./finetuned-marian-best", max_length=128):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="PL->EN medical translation demo.")
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE, help="Polish input.")
    parser.add_argument(
        "--model_dir",
        default="./finetuned-marian-best",
        help="Path to the fine-tuned model directory.",
    )
    args = parser.parse_args()

    print("Source (PL):", args.sentence)
    print("Translation (EN):", translate(args.sentence, args.model_dir))


if __name__ == "__main__":
    main()
