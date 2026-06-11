"""
Visualize level sets of a difference of two quadratic forms.

For p = [x, y]^T, the plotted function is:

    f(p) = (p - a)^T A^T A (p - a)
           - beta (p - b)^T Sigma^{-1} (p - b).

The 4x3 panel sweeps beta over logarithmically spaced values, which helps show
how the objective geometry changes as the ellipsoidal penalty around b becomes
more or less important.
"""

# plot_levels_12_betas_4x3.py
import numpy as np
import matplotlib.pyplot as plt


def f_on_grid(X, Y, a, b, A, Sigma, beta):
    """
    f([x,y]) = (p-a)^T A^T A (p-a) - beta (p-b)^T Sigma^{-1} (p-b)
    evaluada en malla 2D.
    """
    a = np.asarray(a, float).reshape(2)
    b = np.asarray(b, float).reshape(2)
    A = np.asarray(A, float).reshape(2, 2)
    Sigma = np.asarray(Sigma, float).reshape(2, 2)

    Q1 = A.T @ A
    Q2 = np.linalg.inv(Sigma)

    P = np.stack([X, Y], axis=-1)  # (...,2)
    da = P - a
    db = P - b

    term1 = np.einsum("...i,ij,...j->...", da, Q1, da)
    term2 = np.einsum("...i,ij,...j->...", db, Q2, db)
    return term1 - beta * term2


def main():
    # =====================
    # Parámetros (edítalos)
    # =====================
    a = np.array([0.0, 0.0])
    b = np.array([1.2, -0.6])   # a != b

    A = np.array([[1.3, 0.2],
                  [0.1, 0.9]])

    Sigma = np.array([[1.0, 0.25],
                      [0.25, 1.4]])

    # 12 betas (edita a gusto). Ejemplo: log-espaciados
    betas = np.geomspace(0.01, 10.0, 12)

    # Dominio común
    L = 10.0
    epsilon = 0.15
    n = 520
    xs = np.linspace(-L-epsilon, L+epsilon, n)
    ys = np.linspace(-L-epsilon, L+epsilon, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # =====================
    # Figura 4x3 (grande)
    # =====================
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.ravel()

    for ax, beta in zip(axes, betas):
        F = f_on_grid(X, Y, a, b, A, Sigma, beta)

        # niveles automáticos robustos
        lo = np.percentile(F, 5)
        hi = np.percentile(F, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo, hi = np.min(F), np.max(F)

        levels = np.linspace(lo, hi, 14)

        cs = ax.contour(X, Y, F, levels=levels)
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{v:.2g}")

        ax.scatter([a[0]], [a[1]], s=70, marker="o")
        ax.scatter([b[0]], [b[1]], s=70, marker="^")

        ax.set_title(f"beta = {beta:.3g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
