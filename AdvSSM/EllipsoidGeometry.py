"""
Visualize quadratic feasible regions induced by 2D linear maps.

For each matrix A, the plotted set is:

    R(A) = { x in R^2 : x^T A^T A x <= 1 }.

If A^T A is positive definite, R(A) is a bounded ellipse. If A^T A is rank
deficient, at least one direction is unpenalized and the feasible set becomes
unbounded, for example a strip. This small script is a geometry sanity check
for the ellipsoidal attack regions used elsewhere in the repository.
"""

import numpy as np
import matplotlib.pyplot as plt


def random_orthogonal_2d(seed: int = 0) -> np.ndarray:
    """Return a reproducible 2D orthogonal matrix."""
    rng = np.random.default_rng(seed)
    gaussian_matrix = rng.normal(size=(2, 2))
    orthogonal_matrix, triangular_factor = np.linalg.qr(gaussian_matrix)
    orthogonal_matrix *= np.sign(np.diag(triangular_factor))
    return orthogonal_matrix


def evaluate_quadratic_form_on_grid(matrix: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Evaluate [x, y] M [x, y]^T on a 2D meshgrid."""
    return matrix[0, 0] * grid_x**2 + 2 * matrix[0, 1] * grid_x * grid_y + matrix[1, 1] * grid_y**2


def build_demo_transforms() -> list[tuple[str, np.ndarray]]:
    """Create identity, full-rank, and rank-deficient transforms."""
    identity_transform = np.eye(2)

    full_rank_left_rotation = random_orthogonal_2d(seed=1)
    full_rank_right_rotation = random_orthogonal_2d(seed=2)
    full_rank_singular_values = np.diag([2.0, 0.5])
    full_rank_transform = full_rank_left_rotation @ full_rank_singular_values @ full_rank_right_rotation.T

    rank_deficient_left_rotation = random_orthogonal_2d(seed=3)
    rank_deficient_right_rotation = random_orthogonal_2d(seed=4)
    rank_deficient_singular_values = np.diag([-2.0, 0.0])
    rank_deficient_transform = (
        rank_deficient_left_rotation @ rank_deficient_singular_values @ rank_deficient_right_rotation.T
    )

    return [
        ("A = I  (unit circle)", identity_transform),
        ("A full-rank, non-diagonal  (rotated ellipse)", full_rank_transform),
        ("A rank-deficient, non-diagonal  (unbounded strip)", rank_deficient_transform),
    ]


def main() -> None:
    limit = 3.0
    grid_size = 600
    axis_values = np.linspace(-limit, limit, grid_size)
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (title, transform_matrix) in zip(axes, build_demo_transforms()):
        quadratic_matrix = transform_matrix.T @ transform_matrix
        quadratic_values = evaluate_quadratic_form_on_grid(quadratic_matrix, grid_x, grid_y)

        ax.contourf(grid_x, grid_y, (quadratic_values <= 1.0).astype(float), levels=[-0.5, 0.5, 1.5])
        ax.contour(grid_x, grid_y, quadratic_values, levels=[1.0])

        eigenvalues = np.linalg.eigvalsh(quadratic_matrix)
        rank = np.sum(eigenvalues > 1e-10)

        ax.set_title(f"{title}\nrank(A^T A)={rank}, eig={np.round(eigenvalues, 3)}")
        ax.set_aspect("equal", "box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
