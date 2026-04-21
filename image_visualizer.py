"""
image_visualizer.py
===================
Fungsi visualisasi untuk sinyal 2D dan pengolahan gambar.
Menggunakan matplotlib (konsisten dengan visualizer.py).
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# Konsisten dengan visualizer.py
STYLE = {
    "continuous_color" : "#2196F3",
    "discrete_color"   : "#F44336",
    "freq_color"       : "#9C27B0",
    "conv_color"       : "#4CAF50",
    "kernel_color"     : "#FF9800",
    "bg_color"         : "#FAFAFA",
    "grid_alpha"       : 0.3,
}

plt.rcParams.update({
    "figure.facecolor" : STYLE["bg_color"],
    "axes.facecolor"   : "white",
    "axes.grid"        : False,   # grid dimatikan untuk gambar 2D
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


def _matrix_to_display(image):
    """Konversi matrix 2D (list of list) ke format yang bisa diplot matplotlib."""
    return image  # matplotlib imshow bisa langsung terima list of list


def _clamp_matrix(image, lo=0.0, hi=255.0):
    """Clamp semua nilai matrix ke rentang lo–hi."""
    return [[max(lo, min(hi, v)) for v in row] for row in image]


# ─────────────────────────────────────────────
#  1. TAMPILKAN SATU GAMBAR 2D
# ─────────────────────────────────────────────

def plot_image(image, title="Gambar", cmap="gray", show=True):
    """Tampilkan satu gambar grayscale 2D."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255, aspect="equal")
    ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────
#  2. TAMPILKAN GAMBAR SINTETIS DASAR
# ─────────────────────────────────────────────

def plot_all_synthetic_images(images_data):
    """
    Tampilkan beberapa gambar sintetis sekaligus.

    Parameter:
        images_data : list of dict {'image': matrix2D, 'title': str}
    """
    n = len(images_data)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle("Sinyal 2D Sintetis — Gambar Dasar", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                ax = axes[r][c]
                ax.imshow(images_data[idx]["image"], cmap="gray", vmin=0, vmax=255)
                ax.set_title(images_data[idx]["title"], fontsize=10, fontweight="bold")
                ax.axis("off")
            else:
                axes[r][c].set_visible(False)
            idx += 1

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  3. VISUALISASI KONVOLUSI 2D
# ─────────────────────────────────────────────

def plot_convolution2d(original, kernel, result, kernel_name="Kernel", filter_name=""):
    """
    Tampilkan pipeline konvolusi 2D:
    Gambar Asli | Kernel (heatmap) | Hasil Konvolusi
    """
    from image_signals import normalize_matrix, clamp

    result_clamped = [[max(0.0, min(255.0, v)) for v in row] for row in result]
    result_norm    = normalize_matrix(result)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Konvolusi 2D — {filter_name}", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # ① Gambar asli
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("① Gambar Asli  f(x,y)", fontsize=11)
    ax1.axis("off")

    # ② Kernel
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(kernel, cmap="hot", aspect="equal")
    ax2.set_title(f"② Kernel  h(x,y)\n{kernel_name}", fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    # Anotasi nilai kernel
    kh, kw = len(kernel), len(kernel[0])
    if kh <= 7 and kw <= 7:
        for ki in range(kh):
            for kj in range(kw):
                ax2.text(kj, ki, f"{kernel[ki][kj]:.2f}",
                         ha="center", va="center",
                         fontsize=7, color="white" if kernel[ki][kj] < 0.5 else "black")

    # ③ Hasil konvolusi
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(result_norm, cmap="gray", vmin=0, vmax=255)
    ax3.set_title("③ Hasil  g(x,y) = f * h", fontsize=11)
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_convolution2d_comparison(original, results_data):
    """
    Bandingkan beberapa hasil konvolusi dengan kernel berbeda.

    Parameter:
        original     : matrix 2D gambar asli
        results_data : list of dict {'result': matrix2D, 'title': str}
    """
    from image_signals import normalize_matrix

    n = len(results_data) + 1  # +1 untuk gambar asli
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Perbandingan Berbagai Kernel Konvolusi 2D",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Gambar Asli", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i, data in enumerate(results_data):
        result_norm = normalize_matrix(data["result"])
        axes[i + 1].imshow(result_norm, cmap="gray", vmin=0, vmax=255)
        axes[i + 1].set_title(data["title"], fontsize=10, fontweight="bold")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  4. VISUALISASI FFT 2D
# ─────────────────────────────────────────────

def plot_fft2d_pipeline(image, magnitude_shifted, reconstructed, signal_name=""):
    """
    Tampilkan pipeline FFT 2D lengkap:
    Gambar Asli | Spektrum Magnitude (log) | Rekonstruksi IFFT 2D
    """
    from image_signals import normalize_matrix

    mag_norm  = normalize_matrix(magnitude_shifted)
    rec_clamp = [[max(0.0, min(255.0, v)) for v in row] for row in reconstructed]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FFT 2D Pipeline — {signal_name}", fontsize=13, fontweight="bold")

    # ① Gambar asli
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("① Gambar Asli  f(x,y)", fontsize=11)
    axes[0].axis("off")

    # ② Spektrum magnitude
    axes[1].imshow(mag_norm, cmap="magma", vmin=0, vmax=255)
    axes[1].set_title("② Spektrum Magnitude\n|F(u,v)| — log scale, DC di tengah", fontsize=11)
    axes[1].axis("off")

    # ③ Rekonstruksi
    axes[2].imshow(rec_clamp, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("③ Rekonstruksi IFFT 2D\nf̂(x,y)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_fft2d_filter_pipeline(original, mag_original, filtered, mag_filtered,
                                result, filter_name="Low-Pass"):
    """
    Pipeline filtering domain frekuensi:
    Asli | Spektrum asli | Spektrum terfilter | Hasil
    """
    from image_signals import normalize_matrix

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Filtering Domain Frekuensi — {filter_name} Filter",
                 fontsize=13, fontweight="bold")

    mag_orig_norm = normalize_matrix(mag_original)
    mag_filt_norm = normalize_matrix(mag_filtered)
    result_clamp  = [[max(0.0, min(255.0, v)) for v in row] for row in result]

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("① Gambar Asli", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(mag_orig_norm, cmap="magma")
    axes[1].set_title("② Spektrum Asli\n|F(u,v)|", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(mag_filt_norm, cmap="magma")
    axes[2].set_title(f"③ Setelah {filter_name} Mask\n|F_filtered(u,v)|", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(result_clamp, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("④ Hasil Rekonstruksi\nIFFT 2D", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  5. PLOT NOISE BEFORE/AFTER
# ─────────────────────────────────────────────

def plot_denoising(original, noisy, denoised, method="Gaussian Filter"):
    """Tampilkan proses denoising: Asli | Noisy | Hasil filter."""
    from image_signals import normalize_matrix, clamp

    denoised_norm = normalize_matrix(denoised)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Denoising dengan {method}", fontsize=13, fontweight="bold")

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("① Gambar Asli", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("② + Gaussian Noise", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(denoised_norm, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"③ Setelah {method}", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
