import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def random_orthogonal_2d(seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))  # estabiliza signos
    return Q

def quad_form_grid(M, X, Y):
    # para cada punto (x,y): [x y] M [x y]^T
    return M[0,0]*X**2 + 2*M[0,1]*X*Y + M[1,1]*Y**2

# -----------------------------
# Define A1, A2, A3 (A2 y A3 NO diagonales)
# -----------------------------
A1 = np.eye(2)

U2 = random_orthogonal_2d(seed=1)
V2 = random_orthogonal_2d(seed=2)
Sigma2 = np.diag([2.0, 0.5])         # full-rank -> elipse cerrada
A2 = U2 @ Sigma2 @ V2.T              # no diagonal en general

U3 = random_orthogonal_2d(seed=3)
V3 = random_orthogonal_2d(seed=4)
Sigma3 = np.diag([-2.0, 0.0])         # rango 1 -> región no acotada (banda)
A3 = U3 @ Sigma3 @ V3.T              # no diagonal en general

As = [A1, A2, A3]
titles = [
    "A1 = I  (círculo)",
    "A2 no diagonal, full-rank  (elipse rotada)",
    "A3 no diagonal, rank-deficient  (región no acotada: banda)"
]

# -----------------------------
# Grid + plot region x^T A^T A x <= 1
# -----------------------------
lim = 3.0
n = 600
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

for ax, A, title in zip(axes, As, titles):
    M = A.T @ A
    Z = quad_form_grid(M, X, Y)

    # Región interior: Z <= 1 (relleno)
    ax.contourf(X, Y, (Z <= 1.0).astype(float), levels=[-0.5, 0.5, 1.5])

    # Borde: Z = 1
    ax.contour(X, Y, Z, levels=[1.0])

    eigs = np.linalg.eigvalsh(M)
    rank = np.sum(eigs > 1e-10)

    ax.set_title(f"{title}\nrank(A^T A)={rank}, eig={np.round(eigs, 3)}")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

plt.tight_layout()
plt.show()

print("A2=\n", A2)
print("\nA3=\n", A3)
