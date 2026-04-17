"""Challenge metric: harmonic mean of avg per-sample CER on inD and ooD speakers."""
from jiwer import cer


def _average_cer(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    return 100.0 * sum(cer(r, p) for r, p in zip(refs, preds)) / len(preds)


def _harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) else 0.0


def compute_score(
    predictions: list[str],
    references: list[str],
    is_in_domain: list[bool],
) -> dict[str, float]:
    """Returns {'ind_cer', 'ood_cer', 'score'} in %."""
    assert len(predictions) == len(references) == len(is_in_domain)

    ind_predictions: list[str] = []
    ind_references: list[str] = []
    ood_predictions: list[str] = []
    ood_references: list[str] = []
    for prediction, reference, in_domain in zip(predictions, references, is_in_domain):
        if in_domain:
            ind_predictions.append(prediction)
            ind_references.append(reference)
        else:
            ood_predictions.append(prediction)
            ood_references.append(reference)

    ind_cer = _average_cer(ind_predictions, ind_references)
    ood_cer = _average_cer(ood_predictions, ood_references)
    return {
        "ind_cer": ind_cer,
        "ood_cer": ood_cer,
        "score": _harmonic_mean(ind_cer, ood_cer),
    }
