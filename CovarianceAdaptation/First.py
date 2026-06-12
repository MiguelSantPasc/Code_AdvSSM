import numpy as np
import matplotlib.pyplot as plt


def covariance_ellipse_points(mean, cov, n_std=2.0, num_points=400):
    """
    Devuelve los puntos de la elipse asociada a una gaussiana 2D:
        (x - mean)^T cov^{-1} (x - mean) = n_std^2

    Parameters
    ----------
    mean : array-like of shape (2,)
        Centro de la elipse.
    cov : array-like of shape (2, 2)
        Matriz de covarianza simétrica definida positiva.
    n_std : float
        Radio en unidades de desviación típica (por ejemplo 1, 2, 3).
    num_points : int
        Número de puntos para discretizar la elipse.

    Returns
    -------
    pts : ndarray of shape (num_points, 2)
        Puntos de la elipse.
    """
    mean = np.asarray(mean, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)

    # Autovalores/autovectores
    eigvals, eigvecs = np.linalg.eigh(cov)

    if np.any(eigvals <= 0):
        raise ValueError("La matriz de covarianza debe ser definida positiva.")

    # Circunferencia unitaria
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])  # (2, num_points)

    # Transformación lineal para obtener la elipse
    # cov = Q diag(eigvals) Q^T
    # elipse = mean + n_std * Q diag(sqrt(eigvals)) circle
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    pts = mean[:, None] + n_std * transform @ circle

    return pts.T


def plot_original_and_updated_ellipses(
    mean,
    V,
    u,
    lam,
    n_std=2.0,
    num_points=400,
    title="Elipses antes y después del cambio en una dirección"
):
    """
    Dibuja las elipses asociadas a V y V' = V + lam * u u^T en R^2.
    """
    mean = np.asarray(mean, dtype=float).reshape(2)
    V = np.asarray(V, dtype=float).reshape(2, 2)
    u = np.asarray(u, dtype=float).reshape(2)

    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        raise ValueError("El vector u no puede ser nulo.")
    u = u / norm_u  # normalizamos

    V_new = V + lam * np.outer(u, u)

    # Puntos de las elipses
    ellipse_old = covariance_ellipse_points(mean, V, n_std=n_std, num_points=num_points)
    ellipse_new = covariance_ellipse_points(mean, V_new, n_std=n_std, num_points=num_points)

    # Para escalar la flecha de u de forma razonable
    scale = n_std * np.sqrt(np.max(np.linalg.eigvalsh(V_new))) * 1.2

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(ellipse_old[:, 0], ellipse_old[:, 1], label="Elipse original", linewidth=2)
    ax.plot(ellipse_new[:, 0], ellipse_new[:, 1], label="Elipse actualizada", linewidth=2, linestyle="--")

    # Centro
    ax.scatter(mean[0], mean[1], s=50, label="Centro")

    # Dirección u
    ax.arrow(
        mean[0], mean[1],
        scale * u[0], scale * u[1],
        head_width=0.08 * scale,
        length_includes_head=True
    )
    ax.text(
        mean[0] + 1.05 * scale * u[0],
        mean[1] + 1.05 * scale * u[1],
        "u",
        fontsize=12
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    return V_new


if __name__ == "__main__":
    # Centro de la gaussiana
    mean = np.array([0.0, 0.0])

    # Covarianza original
    V = np.array([
        [2.0, 0.6],
        [0.6, 1.0]
    ])

    # Dirección en la que quieres aumentar varianza
    u = np.array([1.0, 1.0])  # se normaliza dentro de la función

    # Intensidad del cambio
    lam = 2.0

    V_new = plot_original_and_updated_ellipses(
        mean=mean,
        V=V,
        u=u,
        lam=lam,
        n_std=2.0,
        title="Comparación de elipses: V vs V + λ u u^T"
    )

    print("Covarianza original V:")
    print(V)
    print("\nCovarianza actualizada V':")
    print(V_new)