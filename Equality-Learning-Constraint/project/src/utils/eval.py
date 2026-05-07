import numpy as np
import torch

def compute_test_scores(results, archs, loss_variants, pos_np):
    """
    Calcula una matriu de scores de test per a cada arquitectura i variant de loss.
    Args:
        results: Dict amb (arch_name, loss_name) -> (model, ...)
        archs: Llista de tuples (arch_name, ...)
        loss_variants: Llista de tuples (_, loss_name)
        pos_np: Mostres positives (N,2)
    Retorna:
        test_scores: np.ndarray de shape (len(archs), len(loss_variants))
    """
    test_scores = np.zeros((len(archs), len(loss_variants)), dtype=float)
    for r, (arch_name, *_ ) in enumerate(archs):
        for c, (_, vname) in enumerate(loss_variants):
            key = (arch_name, vname)
            model, _ = results[key]
            pos = torch.tensor(pos_np, dtype=torch.float32)
            score = 1.0 - float(torch.abs(model(pos)).mean().item())
            test_scores[r, c] = score
    test_scores = np.array(test_scores, dtype=float)
    assert np.all(np.isfinite(test_scores)), "test_scores conté valors no finits!"
    return test_scores

def eval_accuracy_simple(model, pos_test, neg_test, pos_eps=0.04, neg_thr=0.08):
    """Simple accuracy metrics with tolerances.

    Interprets positive points as correct when |h(x)| <= pos_eps.
    Interprets negative points as correct when |h(x)| >= neg_thr.
    Returns both rates and absolute correct counts.
    """
    with torch.no_grad():
        h_pos = model(pos_test).abs()
        h_neg = model(neg_test).abs()

        pos_mask = (h_pos <= pos_eps)
        neg_mask = (h_neg >= neg_thr)

        pos_correct = float(pos_mask.float().mean().item())
        neg_correct = float(neg_mask.float().mean().item())
        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        global_acc = (pos_correct + neg_correct) / 2.0

    return {
        'pos_accuracy': pos_correct,
        'neg_accuracy': neg_correct,
        'pos_correct_count': pos_count,
        'neg_correct_count': neg_count,
        'global_accuracy': global_acc,
        'mean_h_pos': float(h_pos.mean().item()),
        'mean_h_neg': float(h_neg.mean().item()),
        'separation': float(h_neg.mean().item() - h_pos.mean().item())
    }
