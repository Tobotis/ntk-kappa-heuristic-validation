"""Gradient extraction for token-level NTK embeddings."""

from collections.abc import Callable

import torch
from tqdm.auto import tqdm


def extract_grad(model: torch.nn.Module, inputs: dict[str, torch.Tensor], backward_fn: Callable[[torch.Tensor], None], keep_on_gpu: bool = False) -> torch.Tensor:
    """Forward + backward, return flattened gradient for all trainable params (float32)."""
    model.zero_grad()
    out = model(**inputs)
    backward_fn(out.logits[0, -1])  # backward on the last logit (shape: vocab_size)
    grads = [] 
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    model.zero_grad()
    result = torch.cat(grads).float()
    return result if keep_on_gpu else result.cpu()


# ---------------------------------------------------------------------------
# Memory-efficient Gram matrix via layer-by-layer accumulation
# ---------------------------------------------------------------------------
def _get_param_groups(model):
    """Return list of (group_name, [param_names]) for trainable params,
    grouped by transformer layer. Non-layer params (embed, head) each
    get their own group."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # group by transformer layer index, e.g. "layers.5.xxx" → "layers.5"
        parts = name.split(".")
        try:
            layer_idx = next(i for i, tok in enumerate(parts) if tok == "layers")
            key = ".".join(parts[: layer_idx + 2])  # e.g. "model.layers.5"
        except StopIteration:
            key = name  # embed / lm_head / etc.
        groups.setdefault(key, []).append(name)
    return list(groups.items())


def _extract_group_grad(model, param_names, param_lookup):
    """Extract and concat gradients for a specific parameter group."""
    grads = []
    for name in param_names:
        p = param_lookup[name]
        if p.grad is not None:
            grads.append(p.grad.detach().flatten().float())
        else:
            grads.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
    return torch.cat(grads)


def compute_gram_layerwise(model, inputs, token_list, token_ids,
                           max_chunk_numel=10_000_000):
    """Compute Gram matrix G and x_all projection, accumulating over param groups.

    Groups param-groups into batches that fit within max_chunk_numel total
    parameters, then runs one set of (n+1) backward passes per batch.
    Within each backward pass, extracts gradients for all groups in the batch.

    For single-layer mode (1 group), this is one round of backward passes.
    For Lall mode (~24 groups), groups are batched to avoid redundant work.
    """
    n = len(token_list)
    param_groups = _get_param_groups(model)
    param_lookup = dict(model.named_parameters())

    G = torch.zeros(n, n)       # accumulator, CPU
    g_all_vec = torch.zeros(n)  # accumulator, CPU

    model.eval()

    # --- Batch param groups to amortize backward passes ---
    # Each batch holds groups whose combined numel fits in max_chunk_numel.
    group_batches = []
    current_groups, current_batch_numel = [], 0
    for group_name, param_names in param_groups:
        group_numel = sum(param_lookup[param_name].numel() for param_name in param_names) # num of eleents in param group
        if current_groups and current_batch_numel + group_numel > max_chunk_numel:
            # start a new batch
            group_batches.append(current_groups) 
            current_groups, current_batch_numel = [], 0
        # extend groups by current group
        current_groups.append((group_name, param_names, group_numel))
        current_batch_numel += group_numel
    if current_groups:
        group_batches.append(current_groups)

    for group_batch in tqdm(group_batches, desc="Gram (batched)"):
        # One set of backward passes for all groups in this batch
        model.zero_grad()
        out = model(**inputs)
        logits = out.logits[0, -1]

        # x_all gradients for each group in batch
        torch.logsumexp(logits, dim=0).backward(retain_graph=True)
        batch_x_all = [
            _extract_group_grad(model, param_names, param_lookup).cpu()
            for _, param_names, _ in group_batch
        ]

        # Per-token gradients for each group in batch
        batch_X = [torch.zeros(n, group_numel) for _, _, group_numel in group_batch]
        for i, ts in enumerate(token_list):
            model.zero_grad()
            logits[token_ids[ts]].backward(retain_graph=(i < n - 1))
            for group_index, (_, param_names, _) in enumerate(group_batch):
                batch_X[group_index][i] = _extract_group_grad(model, param_names, param_lookup).cpu()

        del out, logits

        # Accumulate contributions from each group
        for group_index in range(len(group_batch)):
            G += batch_X[group_index] @ batch_X[group_index].T
            g_all_vec += batch_X[group_index] @ batch_x_all[group_index]

        del batch_X, batch_x_all

    model.zero_grad()
    return G, g_all_vec


