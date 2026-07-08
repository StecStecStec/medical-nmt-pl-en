"""Metric computation and result persistence for the medical PL->EN NMT project.

Computes BLEU (evaluate/sacreBLEU) and chrF (sacreBLEU CHRF, better suited to
morphologically rich / medical-domain MT). Also writes the per-epoch training
log and the baseline-vs-fine-tuned results file so scores are permanently
recorded rather than only printed to stdout.

Scales:
  - BLEU: float in [0, 1]
  - chrF: float in [0, 100]
"""

import json

import numpy as np
import evaluate

_bleu = None
_chrf = None


def _lazy_load():
    global _bleu, _chrf
    if _bleu is None:
        _bleu = evaluate.load("bleu")
    if _chrf is None:
        _chrf = evaluate.load("chrf")


def decode_preds_labels(predictions, labels, tokenizer):
    """Decode model predictions and gold labels into stripped strings."""
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds_arr = np.array(predictions)
    if preds_arr.ndim == 3:  # logits -> token ids
        preds_arr = np.argmax(preds_arr, axis=-1)

    labels_arr = np.array(labels)
    labels_arr = np.where(labels_arr != -100, labels_arr, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds_arr, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_arr, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]
    return decoded_preds, decoded_labels


def compute_bleu_chrf(decoded_preds, decoded_labels):
    """Compute BLEU and chrF for decoded strings (single reference each)."""
    _lazy_load()
    references = [[lab] for lab in decoded_labels]
    bleu = _bleu.compute(predictions=decoded_preds, references=references)["bleu"]
    chrf = _chrf.compute(predictions=decoded_preds, references=references)["score"]
    return {"bleu": bleu, "chrf": chrf}


def build_compute_metrics(tokenizer):
    """Return a Trainer-compatible compute_metrics closure (BLEU + chrF)."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds, decoded_labels = decode_preds_labels(
            predictions, labels, tokenizer
        )
        return compute_bleu_chrf(decoded_preds, decoded_labels)

    return compute_metrics


def save_training_log(log_history, path):
    """Persist per-epoch train loss + eval loss/BLEU/chrF and full log history."""
    eval_entries = [e for e in log_history if "eval_bleu" in e]
    train_entries = [e for e in log_history if "loss" in e and "epoch" in e]

    per_epoch = []
    for ev in eval_entries:
        ep = ev["epoch"]
        epoch_losses = [t["loss"] for t in train_entries if ep - 1 < t["epoch"] <= ep]
        per_epoch.append(
            {
                "epoch": ep,
                "mean_train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "eval_loss": ev.get("eval_loss"),
                "eval_bleu": ev.get("eval_bleu"),
                "eval_chrf": ev.get("eval_chrf"),
            }
        )

    out = {"per_epoch": per_epoch, "log_history": log_history}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def save_results(baseline_metrics, finetuned_metrics, path, model_info=None):
    """Write baseline vs fine-tuned BLEU/chrF (and deltas) to a JSON file."""

    def pick(m):
        return {"bleu": m.get("eval_bleu"), "chrf": m.get("eval_chrf")}

    b = pick(baseline_metrics)
    f = pick(finetuned_metrics)

    def delta(key):
        if b[key] is None or f[key] is None:
            return None
        return f[key] - b[key]

    results = {
        "model": model_info or {},
        "baseline": b,
        "finetuned": f,
        "delta": {"bleu": delta("bleu"), "chrf": delta("chrf")},
        "notes": "BLEU in [0,1]; chrF in [0,100] (sacreBLEU CHRF).",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    return results
