"""Model loading, freezing, and weight management for Qwen gradient experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

def load_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    layer_id: int | list[int] | None = 23,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load model, freeze all except specified layer(s), return (model, tokenizer, device)."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    ).to(device)

    if layer_id is not None:
        layer_ids = [layer_id] if isinstance(layer_id, int) else list(layer_id)
        freeze_for_layers(model, layer_ids)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

    return model, tokenizer, device


def freeze_for_layers(model: PreTrainedModel, layer_ids: list[int]) -> None:
    """Freeze all params, unfreeze only the specified transformer layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    prefixes = tuple(f"layers.{lid}." for lid in layer_ids)
    for name, p in model.named_parameters():
        if any(pf in name for pf in prefixes):
            p.requires_grad_(True)


def cache_weights(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """Cache pretrained trainable weights (CPU) for fast resets."""
    return {
        name: p.data.clone().cpu()
        for name, p in model.named_parameters() if p.requires_grad
    }


def reset_weights(model: PreTrainedModel, cached: dict[str, torch.Tensor], device: str) -> None:
    """Restore cached pretrained weights for trainable params."""
    for name, p in model.named_parameters():
        if name in cached:
            p.data.copy_(cached[name].to(device))


def get_pi(model: PreTrainedModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return softmax distribution π over vocab at last position."""
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        return torch.softmax(logits, dim=0)
