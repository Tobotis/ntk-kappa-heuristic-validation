import argparse
import math
import time
from pathlib import Path
import torch
from tqdm.auto import tqdm

from model import load_model, cache_weights, reset_weights, get_pi
from grads import extract_grad, compute_gram_layerwise
from kappa import sweep_pairs_vectorized
from exact_pg import run_exact_pg

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

PROMPTS = {
    "movie": "The movie was",
    "food": "The food tasted",
    "weather": "The weather was",
}


def resolve_top_k(tokenizer, pi: torch.Tensor, k: int = 100) -> tuple[list[str], dict[str, dict]]:
    """Select top-k tokens from π, filtered to clean single words.

    Filters:
      - Token text starts with a space (real word boundary)
      - After the space, purely alphabetic (no punctuation / digits / subword junk)
      - At least 2 chars after the space (skip single-letter words like " a", " I")
    """
    topk_probs, topk_ids = torch.topk(pi, k)
    ranks = torch.argsort(pi, descending=True).argsort()

    tokens, token_info = [], {}
    for prob, tid in zip(topk_probs, topk_ids):
        tid = tid.item()
        tok_str = tokenizer.decode(tid)
        # Must start with space and be purely alphabetic after
        if not tok_str.startswith(" "):
            continue
        word = tok_str[1:]
        if len(word) < 2 or not word.isalpha():
            continue
        if tok_str in token_info:
            continue
        tokens.append(tok_str)
        token_info[tok_str] = {"id": tid, "prob": prob.item(), "rank": ranks[tid].item()}
    return tokens, token_info

# - filter the informative pairs
def compute_informative(all_pairs, boundary_thresh=0.0):
    """Select pairs where κ predicts a reversal of the initial majority.
    Two types:
      - 1/2 < rho < κ: a leads but κ predicts b overtakes
      - κ < rho < 1/2: b leads but κ predicts a overtakes
    """
    informative = [p for p in all_pairs
                   if (p["rho"] > 0.5 and p["kappa"] > p["rho"]) or
                      (p["rho"] < 0.5 and p["kappa"] < p["rho"])]
    if boundary_thresh > 0:
        selected = {id(p) for p in informative}
        near = [p for p in all_pairs
                if id(p) not in selected
                and abs(p["kappa"] - p["rho"]) < boundary_thresh]
        informative += near
    informative.sort(key=lambda p: -abs(p["kappa"] - p["rho"]))
    return informative


def select_binned(all_pairs, n_pairs=100, n_bins=10, seed=42):
    """Select pairs with uniform coverage over the (ρ, κ) plane.

    Bins the 2D (ρ, κ) space into n_bins × n_bins cells and draws
    ceil(n_pairs / n_occupied_bins) samples per occupied bin (shuffled),
    then trims to exactly n_pairs.

    Pairs with extreme κ (outside [0, 1]) are clipped into the edge bins
    so they still get represented.
    """
    import random
    rng = random.Random(seed)

    # Bin assignment: clip κ into [0, 1] for binning, ρ is already in (0, 1)
    bins = {}  # (ri, ki) -> list of pairs
    for p in all_pairs:
        ri = min(int(p["rho"] * n_bins), n_bins - 1)
        kappa_clipped = max(0.0, min(1.0, p["kappa"]))
        ki = min(int(kappa_clipped * n_bins), n_bins - 1)
        bins.setdefault((ri, ki), []).append(p)

    # Shuffle within each bin
    for key in bins:
        rng.shuffle(bins[key])

    n_occupied = len(bins)
    per_bin = math.ceil(n_pairs / n_occupied) if n_occupied > 0 else 0

    selected = []
    for key in sorted(bins):  # deterministic iteration order
        selected.extend(bins[key][:per_bin])

    # Trim to n_pairs (if rounding gave us extras)
    rng.shuffle(selected)
    selected = selected[:n_pairs]
    selected.sort(key=lambda p: -abs(p["kappa"] - p["rho"]))
    return selected


def _param_delta(model, w0, device):
    """Compute ||w_T - w_0|| for trainable params."""
    delta_sq = torch.tensor(0.0, device=device)
    for name, p in model.named_parameters():
        if name in w0:
            delta_sq += (p.data - w0[name].to(device)).pow(2).sum()
    return delta_sq.sqrt().item()


def _pair_gram(model, inputs, a_id, b_id):
    """Compute the 2x2 NTK Gram sub-matrix for a token pair given their vocab IDs."""
    x_a = extract_grad(
        model, inputs, lambda logits, t=a_id: logits[t].backward(), keep_on_gpu=True)
    x_b = extract_grad(
        model, inputs, lambda logits, t=b_id: logits[t].backward(), keep_on_gpu=True)
    x_all = extract_grad(
        model, inputs, lambda logits: torch.logsumexp(logits, dim=0).backward(), keep_on_gpu=True)

    return {
        "G_aa": torch.dot(x_a, x_a).item(),
        "G_ab": torch.dot(x_a, x_b).item(),
        "G_bb": torch.dot(x_b, x_b).item(),
        "g_all_a": torch.dot(x_a, x_all).item(),
        "g_all_b": torch.dot(x_b, x_all).item(),
    }


def _gram_drift(pre_gram, post_gram):
    """Relative Frobenius norm change of the 2x2 sub-Gram matrix."""
    pre = torch.tensor([[pre_gram["G_aa"], pre_gram["G_ab"]],
                        [pre_gram["G_ab"], pre_gram["G_bb"]]])
    post = torch.tensor([[post_gram["G_aa"], post_gram["G_ab"]],
                         [post_gram["G_ab"], post_gram["G_bb"]]])
    return (post - pre).norm().item() / pre.norm().item()


def _print_pair_result(pair, rho_f, total_mass, param_delta, gram_drift, is_correct, log):
    """Print one-line summary for a single pair evaluation."""
    mark = "✓" if is_correct else "✗"
    margin = pair["kappa"] - pair["rho"]
    mass_change = total_mass[-1] - total_mass[0]
    drift_str = f"  ΔG={gram_drift:.4f}" if gram_drift is not None else ""
    top3_str = ", ".join(f"{t} ({p:.3f})" for t, p in log["top3"])
    print(f"{mark}  ρ₀={pair['rho']:.4f} κ={pair['kappa']:.4f} m={margin:+.4f} → ρ_f={rho_f:.4f}  "
          f"Σπ: {total_mass[0]:.4f}→{total_mass[-1]:.4f} (Δ={mass_change:+.4f})  "
          f"‖Δw‖={param_delta:.4f}{drift_str}  "
          f"({pair['a'].strip()} vs {pair['b'].strip()})")
    print(f"   top3: {top3_str}")


def _print_accuracy_breakdown(validation, n_correct):
    """Print overall accuracy and per-quadrant breakdown."""
    acc = n_correct / len(validation) if validation else 0
    q_above = [v for v in validation if v["rho"] > 0.5]   # a leads
    q_below = [v for v in validation if v["rho"] <= 0.5]   # b leads
    n_above = sum(v["correct"] for v in q_above) if q_above else 0
    n_below = sum(v["correct"] for v in q_below) if q_below else 0
    print(f"  Accuracy: {n_correct}/{len(validation)} ({100*acc:.1f}%)")
    if q_above:
        print(f"    ρ>½ (κ>ρ, b should overtake a): {n_above}/{len(q_above)} ({100*n_above/len(q_above):.1f}%)")
    if q_below:
        print(f"    ρ<½ (κ<ρ, a should overtake b): {n_below}/{len(q_below)} ({100*n_below/len(q_below):.1f}%)")


def _save_results(args, prompt_name, prompt, lr, model_short,
                  acc, all_pairs, informative, validation):
    """Serialize validation results to a .pt file, return the save path."""
    lid_str = "all" if args.layer_id is None else "_".join(str(l) for l in args.layer_id)
    tag = f"{model_short}_L{lid_str}_{prompt_name}_{args.optimizer}_lr{lr}_T{args.n_steps}"
    save_path = RESULTS / f"validation_{tag}_{int(time.time())}.pt"
    torch.save({
        "config": {
            "model": args.model, "layer_id": args.layer_id,
            "prompt_name": prompt_name, "prompt": prompt,
            "n_steps": args.n_steps, "lr": lr,
            "optimizer": args.optimizer, "rho_thresh": args.rho_thresh,
            "max_prob_ratio": args.max_prob_ratio,
            "boundary_thresh": args.boundary_thresh,
            "pair_selection": args.pair_selection,
            "n_bins": args.n_bins,
            "top_k": args.top_k,
        },
        "accuracy": acc,
        "n_pairs": len(all_pairs),
        "n_informative": len(informative),
        "pairs": [{
            "a": v["a"], "b": v["b"],
            "rho": v["rho"], "kappa": v["kappa"], "pred": v["pred"],
            "rho_final": v["rho_final"], "actual": v["actual"],
            "correct": v["correct"],
            "log": v["log"], "total_mass": v["total_mass"],
            "param_delta": v["param_delta"],
            "post_gram": v["post_gram"],
            "gram_drift": v["gram_drift"],
        } for v in validation],
    }, save_path)
    return save_path


def run_sweep(model, tokenizer, device, inputs, w0,
              test_pairs, n_steps, lr, optimizer_name, rho_thresh,
              pre_G=None, tokens=None):
    """Run PG on all test pairs, return (validation_list, n_correct).
    
    If pre_G/tokens are provided, computes Gram drift per pair.
    """

    validation, correct = [], 0
    pbar = tqdm(test_pairs, desc=f"lr={lr} {optimizer_name}")
    for pair in pbar:
        reset_weights(model, w0, device)
        log = run_exact_pg(model, inputs, tokenizer, [pair["a"]], [pair["b"]], device,
                            n_steps=n_steps, lr=lr, optimizer=optimizer_name,
                            rho_thresh=rho_thresh,verbose=False)

        # Post-training diagnostics (before weight reset)
        param_delta = _param_delta(model, w0, device)
        a_id = tokenizer.encode(pair["a"], add_special_tokens=False)[0]
        b_id = tokenizer.encode(pair["b"], add_special_tokens=False)[0]
        post_gram = _pair_gram(model, inputs, a_id, b_id)
        if pre_G is not None and tokens is not None:
            ai, bi = tokens.index(pair["a"]), tokens.index(pair["b"])
            pre_gram = {
                "G_aa": pre_G[ai, ai].item(), "G_ab": pre_G[ai, bi].item(),
                "G_bb": pre_G[bi, bi].item(),
            }
            gram_drift = _gram_drift(pre_gram, post_gram)
        else:
            gram_drift = None

        rho_i, rho_f = log["rho"][0], log["rho"][-1]
        change_tol = 1e-4
        winner_tol = 1e-6
        if rho_f > rho_i + change_tol:
            actual = "a wins"
        elif rho_f < rho_i - change_tol:
            actual = "b wins"
        elif rho_f > 0.5 + winner_tol:
            actual = "a wins"
        elif rho_f < 0.5 - winner_tol:
            actual = "b wins"
        else:
            actual = "a wins" if rho_i >= 0.5 else "b wins"
        is_correct = actual == pair["pred"]
        correct += is_correct
        total_mass = [a + b for a, b in zip(log["a_mass"], log["b_mass"])]
        validation.append({
            **pair, "rho_final": rho_f, "actual": actual, "correct": is_correct,
            "log": log, "total_mass": total_mass,
            "param_delta": param_delta,
            "post_gram": post_gram,
            "gram_drift": gram_drift,
        })
        _print_pair_result(pair, rho_f, total_mass, param_delta, gram_drift, is_correct, log)
        pbar.set_postfix(acc=f"{correct}/{len(validation)}")

    reset_weights(model, w0, device)
    return validation, correct


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer_id", type=int, nargs="+", default=None)
    parser.add_argument("--prompts", nargs="+", default=["movie"], choices=list(PROMPTS))
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--rho_thresh", type=float, default=0.95)
    parser.add_argument("--max_prob_ratio", type=float, default=100.0)
    parser.add_argument("--boundary_thresh", type=float, default=0.0)
    parser.add_argument("--pair_selection", default="informative",choices=["informative", "dist"])
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model, layer_id=args.layer_id)
    model_short = args.model.split("/")[-1]
    w0 = cache_weights(model)

    for prompt_name in args.prompts:
        prompt = PROMPTS[prompt_name]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"\n[{prompt_name}]")

        # Load cached Gram data if available; otherwise resolve tokens and compute it.
        lid_str = "all" if args.layer_id is None else "_".join(str(l) for l in args.layer_id)
        grad_path = RESULTS / f"grads_{model_short}_L{lid_str}_{prompt_name}_top{args.top_k}.pt"

        if grad_path.exists():
            print(f"  Loading cached Gram matrix from {grad_path.name}")
            grad_data = torch.load(grad_path, weights_only=False)
            G, g_all = grad_data["G"], grad_data["g_all"]
            tokens = grad_data["tokens"]
            token_info = grad_data["token_info"]
            del grad_data
        else:
            pi = get_pi(model, inputs)
            tokens, token_info = resolve_top_k(tokenizer, pi, k=args.top_k)
            tid_map = {tok_str: token_info[tok_str]["id"] for tok_str in tokens}
            G, g_all = compute_gram_layerwise(model, inputs, tokens, tid_map)
            torch.save(
                {"G": G, "g_all": g_all, "tokens": tokens, "token_info": token_info},
                grad_path,
            )
            print(f"  Saved Gram matrix → {grad_path.name}")

        print(f"  {len(tokens)} tokens")

        probs = torch.tensor([token_info[tok_str]["prob"] for tok_str in tokens])
        all_pairs = sweep_pairs_vectorized(tokens, probs, G, g_all,
                                           max_prob_ratio=args.max_prob_ratio)
        if args.pair_selection == "dist":
            informative = select_binned(all_pairs, n_pairs=args.n_pairs,
                                           n_bins=args.n_bins)
            print(f"  {len(all_pairs)} pairs, {len(informative)} selected (binned {args.n_bins}x{args.n_bins})")
        else:
            informative = compute_informative(all_pairs, args.boundary_thresh)
            print(f"  {len(all_pairs)} pairs, {len(informative)} informative")

        # Run PG at each lr
        for lr in args.lr:
            print(f"\n  -- lr={lr}, {args.optimizer}, T={args.n_steps} ──")
            validation, n_correct = run_sweep(
                model, tokenizer, device, inputs, w0,
                informative, args.n_steps, lr, args.optimizer, args.rho_thresh,
                pre_G=G, tokens=tokens)

            acc = n_correct / len(validation) if validation else 0
            _print_accuracy_breakdown(validation, n_correct)
            save_path = _save_results(
                args, prompt_name, prompt, lr, model_short,
                acc, all_pairs, informative, validation)
            print(f"  Saved → {save_path}")

            # Free validation data (large tensors in post_x_* and logits_snapshots)
            del validation

        # Free prompt-level data between prompts
        del inputs, all_pairs, informative


if __name__ == "__main__":
    main()
