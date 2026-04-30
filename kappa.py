"""Compute (ρ₀, κ) for token pairs — single-pair and vectorized via Gram matrix."""

import torch


def sweep_pairs_vectorized(token_list, probs, G, g_all, max_prob_ratio=10.0):
    """Compute (ρ₀, κ) for all qualifying pairs from a precomputed Gram matrix.

    Args:
        token_list: list of token strings (in same order as G)
        probs: tensor of probabilities (same order)
        G: (n, n) Gram matrix
        g_all: (n,) dot products with x_all
        max_prob_ratio: only pair tokens within this probability ratio

    Returns:
        list of dicts with keys: a, b, rho, kappa, pred
    """
    n = len(token_list)
    ii, jj = torch.triu_indices(n, n, offset=1) # grab pair indices
    pa, pb = probs[ii], probs[jj] # grab probabilities for those pairs
    mask = (torch.max(pa, pb) / torch.min(pa, pb)) <= max_prob_ratio
    ii, jj = ii[mask], jj[mask] # only keep those where probability ratio is fine
    pa, pb = probs[ii], probs[jj]

    p_S = pa + pb
    rho = pa / p_S

    xc_xa = (g_all[ii] - pa * G[ii, ii] - pb * G[jj, ii]) / (1 - p_S)
    xc_xb = (g_all[jj] - pa * G[ii, jj] - pb * G[jj, jj]) / (1 - p_S)

    numer = xc_xa - xc_xb - G[ii, jj] + G[jj, jj]
    denom = G[ii, ii] - 2 * G[ii, jj] + G[jj, jj]
    kappa = numer / denom

    results = []
    for k in range(len(ii)):
        i, j = ii[k].item(), jj[k].item()
        r, kap = rho[k].item(), kappa[k].item()
        results.append({
            "a": token_list[i], "b": token_list[j],
            "rho": r, "kappa": kap,
            "pred": "a wins" if r > kap else "b wins",
        })
    return results
