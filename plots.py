"""Plotting utilities for ρ₀/κ experiments."""

import matplotlib.pyplot as plt

def plot_rho_kappa_scatter(results, ax=None, title=None, label_points=False, fontsize=5,
                           color_by_truth=False, label_filter="all",
                           half_range=False):
    """Scatter plot of (ρ₀, κ) with diagonal and shaded regions.

    Parameters
    ----------
    results : list[dict]
        Each dict must have 'rho' and 'kappa'. Optionally 'actual' (for
        colouring), 'correct', 'a'/'pos', 'b'/'neg' (for labels).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None a new figure is created.
    title : str, optional
        Subplot title.
    label_points : bool
        Annotate each point with its token-pair names.
    fontsize : int
        Font size for point annotations.
    color_by_truth : bool
        If True and 'actual' is present, colour points green (a wins) /
        red (b wins) instead of uniform steelblue.
    label_filter : {"all", "incorrect", "none"}
        Which points to annotate when *label_points* is True.
    half_range : bool
        If True, restrict both axes to [0.5, 1] instead of [0, 1].
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    rhos = [r["rho"] for r in results]
    kappas = [r["kappa"] for r in results]

    eps = 0.02
    x_lo = 0.5 - eps if half_range else 0.0 - eps
    x_hi = 1.0 + eps

    # y-axis: default [0, 1], expand if data exceeds
    k_min = min(kappas) if kappas else 0.0
    k_max = max(kappas) if kappas else 1.0
    y_lo = min(0.0 - eps, k_min - eps)
    y_hi = max(1.0 + eps, k_max + eps)

    # Draw shading and diagonal first (behind points)
    diag = [min(x_lo, y_lo), max(x_hi, y_hi)]
    ax.fill_between(diag, diag, diag[1], alpha=0.2, color="red", label=r"$b$ wins region", zorder=0)
    ax.fill_between(diag, diag[0], diag, alpha=0.2, color="green", label=r"$a$ wins region", zorder=0)
    ax.plot(diag, diag, "k--", lw=1, label=r"$\rho_0 = \kappa$", zorder=1)

    if color_by_truth and all("actual" in r for r in results):
        colors = ["#2ca02c" if r["actual"] == "a wins" else "#d62728" for r in results]
        ax.scatter(rhos, kappas, s=30, alpha=0.7, c=colors, edgecolors="k", linewidths=0.5, zorder=2)
    else:
        ax.scatter(rhos, kappas, s=30, alpha=0.7, c="steelblue", edgecolors="k", linewidths=0.5, zorder=2)

    if label_points and label_filter != "none":
        for r in results:
            if label_filter == "incorrect" and r.get("correct", True):
                continue
            a_label = r.get("pos", r.get("a", "")).strip()
            b_label = r.get("neg", r.get("b", "")).strip()
            ax.annotate(f"{a_label}/{b_label}", (r["rho"], r["kappa"]),
                        fontsize=fontsize, alpha=0.7, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 3))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$\rho_0$"); ax.set_ylabel(r"$\kappa$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9); ax.set_box_aspect(1)
    return fig, ax
