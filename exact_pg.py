"""Exact policy gradient loop: track ρ(t) under analytic expected PG."""

import math
import torch

def run_exact_pg(model, inputs, tokenizer, a_tokens, b_tokens, device,
                  n_steps=50, lr=1e-4, optimizer="sgd", rho_thresh=0.95,
                  verbose=False, a_ids_override=None, b_ids_override=None):
    """Exact expected PG with reward=1 for a+b tokens. Returns rho(t) log.

    If a_ids_override / b_ids_override are provided (as tensors of vocab
    indices), they are used directly instead of re-encoding a_tokens/b_tokens.
    This avoids the decode→encode round-trip which is lossy for BPE tokens.
    """
    if a_ids_override is not None and b_ids_override is not None:
        a_ids = a_ids_override.to(device)
        b_ids = b_ids_override.to(device)
        r_ids = torch.cat([a_ids, b_ids])
    else:
        rewarded = a_tokens + b_tokens
        r_ids = torch.tensor([tokenizer.encode(t, add_special_tokens=False)[0] for t in rewarded], device=device)
        a_ids = torch.tensor([tokenizer.encode(t, add_special_tokens=False)[0] for t in a_tokens], device=device)
        b_ids = torch.tensor([tokenizer.encode(t, add_special_tokens=False)[0] for t in b_tokens], device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.SGD(params, lr=lr)

    log = {
        "step": [], "a_mass": [], "b_mass": [], "rho": [], "logit_rho": [],
        "theta_a": [], "theta_b": [], "loss": [], "grad_norm": [],
    }
    model.train()

    for step in range(n_steps):
        opt.zero_grad()
        logits_s = model(**inputs).logits[0, -1]

        lp = torch.log_softmax(logits_s, dim=0)
        pr = torch.softmax(logits_s, dim=0)

        loss = -(pr[r_ids].detach() * lp[r_ids]).sum()
        loss.backward()

        with torch.no_grad():
            # Single sync: gather all scalars on GPU, transfer once
            stats = torch.stack([
                pr[a_ids].sum(),
                pr[b_ids].sum(),
                logits_s[a_ids[0]],
                logits_s[b_ids[0]],
                loss.detach(),
            ])
            pa, pb, theta_a, theta_b, loss_val = stats.tolist()

            # Grad norm: compute infrequently (only for logging)
            if step % 20 == 0:
                gn = sum(p.grad.pow(2).sum() for p in params if p.grad is not None).sqrt().item()
            else:
                gn = float('nan')

        opt.step()

        rho = pa / (pa + pb) if (pa + pb) > 0 else 0.5
        rho_clamp = max(min(rho, 1 - 1e-12), 1e-12)
        log["step"].append(step)
        log["a_mass"].append(pa)
        log["b_mass"].append(pb)
        log["rho"].append(rho)
        log["logit_rho"].append(math.log(rho_clamp / (1 - rho_clamp)))
        log["theta_a"].append(theta_a)
        log["theta_b"].append(theta_b)
        log["loss"].append(loss_val)
        log["grad_norm"].append(gn)

        if verbose:
            reward_mass = pa + pb
            print(f"    step {step+1}/{n_steps}: loss={loss_val:.6f}  "
                  f"reward={reward_mass:.4f}  ρ={rho:.4f}")

        if rho > rho_thresh or rho < (1 - rho_thresh):
            if verbose:
                print(f"    → collapsed at step {step+1} (ρ={rho:.4f})")
            break

    # Store top-3 tokens from the final distribution (no extra forward pass)
    with torch.no_grad():
        top3_probs, top3_ids = torch.topk(pr, 3)
        log["top3"] = [(tokenizer.decode(tid.item()), prob.item())
                       for tid, prob in zip(top3_ids, top3_probs)]

    model.eval()
    return log


