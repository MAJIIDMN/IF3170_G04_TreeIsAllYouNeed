"""
Bonus LR: video garis kontur log-loss dan lintasan parameter (theta0, theta1) selama training.

Model utama punya 100 bobot sehingga permukaan loss-nya tidak bisa digambar di bidang 2D. Karena itu
dilatih model logistic regression dengan tepat dua parameter: theta0 = bias dan theta1 = bobot satu fitur.
Kontur yang digambar adalah permukaan log-loss J(theta0, theta1) yang sebenarnya, bukan irisan, dan
lintasannya adalah parameter yang disimpan kelas LogisticRegression (atribut path) setiap epoch.

Referensi:
- M. L. Khodra, "Logistic Regression: Stochastic Gradient Ascent," YouTube, 2021 (log-loss / LCL,
     update gradien, inisialisasi parameter). https://www.youtube.com/watch?v=O5NSNMQO1yU
- Matplotlib documentation, "matplotlib.pyplot.contour" dan "matplotlib.animation".
     https://matplotlib.org/stable/api/animation_api.html
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

try:   # ffmpeg dari paket imageio-ffmpeg (requirements.txt) untuk menyimpan MP4
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

from algoritma.lr.lr import LogisticRegression

PATH_COLOR = "#d62728"


def train_2param(x, y, init, **kwargs):
    """Latih LR dengan satu fitur x (theta0 = bias, theta1 = bobot x) mulai dari init = (theta0, theta1)."""
    X = np.asarray(x, float).reshape(-1, 1)
    return LogisticRegression(**kwargs).fit(X, y, init=([init[1]], init[0]))


def make_video(x, y, model, init, out_path, fps=12, max_frames=150, grid=90):
    X = np.asarray(x, float).reshape(-1, 1)
    # lintasan (theta0, theta1): titik awal, lalu satu titik per epoch
    path = np.array([init] + [(b, w[0]) for w, b in model.path])
    loss = np.r_[model.loss(X, y, w=np.array([init[1]]), b=init[0]), model.loss_history]

    # grid parameter yang mencakup seluruh lintasan, J dihitung dengan fungsi loss kelas yang sama
    lo, hi = path.min(0), path.max(0)
    pad = 0.25 * (hi - lo).max() + 0.2
    t0 = np.linspace(lo[0] - pad, hi[0] + pad, grid)
    t1 = np.linspace(lo[1] - pad, hi[1] + pad, grid)
    Z = np.array([[model.loss(X, y, w=np.array([b1]), b=b0) for b0 in t0] for b1 in t1])
    levels = loss.min() + (Z.max() - loss.min()) * np.linspace(0.01, 1, 20) ** 2   # rapat di dekat minimum

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax.contourf(t0, t1, Z, levels=levels, cmap="viridis", alpha=0.35)
    cs = ax.contour(t0, t1, Z, levels=levels, cmap="viridis", linewidths=0.8)
    ax.clabel(cs, cs.levels[::4], fmt="%.3f", fontsize=7)
    ax.plot(*init, "kx", markersize=9, mew=2, label=f"titik awal {init}")
    line, = ax.plot([], [], "-", color=PATH_COLOR, linewidth=1.8, label="lintasan parameter")
    dot, = ax.plot([], [], "o", color=PATH_COLOR, markersize=6)
    ax.set_xlabel(r"$\theta_0$ (bias)")
    ax.set_ylabel(r"$\theta_1$ (bobot fitur)")
    ax.legend(fontsize=8, loc="best")

    curve, = ax2.plot([], [], "-", color=PATH_COLOR, linewidth=1.6)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(r"log-loss $J(\theta_0, \theta_1)$")
    ax2.set_xlim(0, len(loss) - 1)
    ax2.set_ylim(loss.min() - 0.02 * (loss.max() - loss.min()), loss.max() * 1.02)
    title = fig.suptitle("")

    def draw(k):
        line.set_data(path[:k + 1, 0], path[:k + 1, 1])
        dot.set_data([path[k, 0]], [path[k, 1]])
        curve.set_data(np.arange(k + 1), loss[:k + 1])
        title.set_text(f"Epoch {k}: $\\theta_0$ = {path[k, 0]:.3f}, $\\theta_1$ = {path[k, 1]:.3f}, "
                       f"J = {loss[k]:.4f}")
        return line, dot, curve, title

    frames = np.unique(np.linspace(0, len(path) - 1, min(max_frames, len(path))).astype(int))
    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg tidak ditemukan. Jalankan: pip install -r requirements.txt (imageio-ffmpeg).")
    out_path = out_path.with_suffix(".mp4")
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
    draw(frames[-1])
    return fig, out_path
