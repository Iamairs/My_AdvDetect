from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _scores(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Scores must be a finite, nonempty 1D array")
    return result


def calibrate_threshold(scores, fpr: float) -> dict:
    """Conservative order-statistic calibration; reject strictly above threshold."""
    values = _scores(scores)
    if not 0 < fpr < 1:
        raise ValueError("Target FPR must be in (0, 1)")
    rank = math.ceil((len(values) + 1) * (1 - fpr))
    threshold = float(np.sort(values)[rank - 1]) if rank <= len(values) else None
    # None encodes +infinity using standards-compliant JSON.
    return {"threshold": threshold, "target_fpr": fpr, "calibration_size": len(values),
            "rank": rank, "reject_rule": "score > threshold",
            "accept_all": threshold is None,
            "score_scale": max(float(values.std()), 1e-8),
            "empirical_calibration_fpr": float((values > threshold).mean()) if threshold is not None else 0.0}


def detection_metrics(clean_scores, adv_scores, clean_predictions, adv_predictions, labels, calibration):
    clean, adv = _scores(clean_scores), _scores(adv_scores)
    labels, cp, ap = map(np.asarray, (labels, clean_predictions, adv_predictions))
    if not (len(clean) == len(adv) == len(labels) == len(cp) == len(ap)):
        raise ValueError("Paired sample counts differ")
    threshold = calibration["threshold"]
    tau = float("inf") if threshold is None else float(threshold)
    correct, wrong = cp == labels, ap != labels
    successful = correct & wrong
    joint = successful & (adv <= tau)
    result = {
        "num_samples": len(labels), "num_clean_correct": int(correct.sum()),
        "num_successful_attacks": int(successful.sum()),
        "clean_accuracy": float(correct.mean()), "adversarial_accuracy": float((~wrong).mean()),
        "attack_success_rate_on_clean_correct": float(successful.sum() / correct.sum()) if correct.any() else None,
        "threshold": threshold, "target_fpr": calibration["target_fpr"],
        "clean_fpr": float((clean > tau).mean()),
        "tpr_on_successful_attacks": float((adv[successful] > tau).mean()) if successful.any() else None,
        "joint_success_rate_all": float(joint.mean()),
        "joint_success_rate_on_clean_correct": float(joint.sum() / correct.sum()) if correct.any() else None,
        "accepted_wrong_rate_all": float((wrong & (adv <= tau)).mean()),
        "attack_escape_rate_on_successful_attacks": float((adv[successful] <= tau).mean()) if successful.any() else None,
    }
    for name, selected in (("all_generated", adv), ("successful_only", adv[successful])):
        if not len(selected):
            result[f"auroc_{name}"] = None
            result[f"aupr_{name}"] = None
            continue
        truth = np.concatenate([np.zeros(len(clean)), np.ones(len(selected))])
        scores = np.concatenate([clean, selected])
        result[f"auroc_{name}"] = float(roc_auc_score(truth, scores))
        result[f"aupr_{name}"] = float(average_precision_score(truth, scores))
    result["paper_protocol_num_clean"] = int(correct.sum())
    result["paper_protocol_num_adv"] = int(successful.sum())
    if correct.any() and successful.any():
        truth = np.concatenate([np.zeros(correct.sum()), np.ones(successful.sum())])
        result["auroc_paper_protocol"] = float(roc_auc_score(truth, np.concatenate([clean[correct], adv[successful]])))
    else:
        result["auroc_paper_protocol"] = None
    return result
