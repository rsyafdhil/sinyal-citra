"""
visualizer.py
=============
Semua fungsi plotting menggunakan matplotlib.
Sinyal dan data diterima dari signals.py, transforms.py, convolution.py.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
#  STYLE GLOBAL
# ─────────────────────────────────────────────

STYLE = {
    "continuous_color" : "#2196F3",   # biru  — sinyal kontinu
    "discrete_color"   : "#F44336",   # merah — sinyal diskrit
    "freq_color"       : "#9C27B0",   # ungu  — domain frekuensi
    "conv_color"       : "#4CAF50",   # hijau — hasil konvolusi
    "kernel_color"     : "#FF9800",   # oranye — kernel
    "bg_color"         : "#FAFAFA",
    "grid_alpha"       : 0.3,
}

plt.rcParams.update({
    "figure.facecolor" : STYLE["bg_color"],
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : STYLE["grid_alpha"],
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


def _add_zero_line(ax):
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)


# ─────────────────────────────────────────────
#  1. PLOT SINYAL DASAR TUNGGAL
# ─────────────────────────────────────────────

def plot_signal(t, x, title="Sinyal", xlabel="n / t", ylabel="Amplitudo",
                discrete=False, color=None, show=True):
    """Plot satu sinyal (kontinu atau diskrit)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    c = color or (STYLE["discrete_color"] if discrete else STYLE["continuous_color"])

    if discrete:
        ax.stem(t, x, linefmt=c, markerfmt="o", basefmt="black")
    else:
        ax.plot(t, x, color=c, linewidth=1.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _add_zero_line(ax)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────
#  2. PLOT SEMUA SINYAL DASAR (1 figure, multi subplot)
# ─────────────────────────────────────────────

def plot_all_basic_signals(signals_data):
    """
    Plot semua sinyal dasar sekaligus.

    Parameter:
        signals_data : list of dict dengan key:
            { 'title', 't', 'x', 'discrete' }
    """
    n = len(signals_data)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle("Sinyal-Sinyal Dasar", fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, data in enumerate(signals_data):
        ax = axes[i]
        t, x = data["t"], data["x"]
        disc = data.get("discrete", False)
        c = STYLE["discrete_color"] if disc else STYLE["continuous_color"]

        if disc:
            ax.stem(t, x, linefmt=c, markerfmt="o", basefmt="black")
        else:
            ax.plot(t, x, color=c, linewidth=1.8)

        ax.set_title(data["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("n" if disc else "t")
        ax.set_ylabel("Amplitudo")
        _add_zero_line(ax)

    # Sembunyikan subplot kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  3. PLOT KONTINU vs DISKRIT (side-by-side)
# ─────────────────────────────────────────────

def plot_continuous_vs_discrete(t_cont, x_cont, n_disc, x_disc,
                                 signal_name="Sinyal Sinus"):
    """Tampilkan sinyal kontinu dan versi diskritnya berdampingan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Dual Domain: {signal_name}", fontsize=13, fontweight="bold")

    # Kontinu
    ax1.plot(t_cont, x_cont, color=STYLE["continuous_color"], linewidth=1.8)
    ax1.set_title("Domain Waktu — Kontinu x(t)")
    ax1.set_xlabel("Waktu (t)")
    ax1.set_ylabel("Amplitudo")
    _add_zero_line(ax1)

    # Diskrit
    ax2.stem(n_disc, x_disc,
             linefmt=STYLE["discrete_color"],
             markerfmt="o",
             basefmt="black")
    ax2.set_title("Domain Waktu — Diskrit x[n]")
    ax2.set_xlabel("Sampel (n)")
    ax2.set_ylabel("Amplitudo")
    _add_zero_line(ax2)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  4. PLOT DUAL DOMAIN (Waktu + Frekuensi)
# ─────────────────────────────────────────────

def plot_dual_domain(t, x, freqs, magnitude, signal_name="Sinyal"):
    """
    Tampilkan sinyal di domain waktu (kiri) dan domain frekuensi (kanan).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"The Dual Domains — {signal_name}", fontsize=13, fontweight="bold")

    # Domain waktu
    ax1.plot(t, x, color=STYLE["continuous_color"], linewidth=1.8)
    ax1.set_title("Domain Waktu  x(t)")
    ax1.set_xlabel("Waktu (t)")
    ax1.set_ylabel("Amplitudo")
    _add_zero_line(ax1)

    # Domain frekuensi
    ax2.bar(freqs, magnitude[:len(freqs)],
            width=freqs[1] - freqs[0] if len(freqs) > 1 else 0.1,
            color=STYLE["freq_color"], alpha=0.8)
    ax2.set_title("Domain Frekuensi  |X(f)|")
    ax2.set_xlabel("Frekuensi (Hz)")
    ax2.set_ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  5. PLOT FFT LENGKAP (Sinyal → FFT → IFFT)
# ─────────────────────────────────────────────

def plot_fft_pipeline(t, x_original, freqs, magnitude, x_reconstructed,
                       signal_name="Sinyal"):
    """
    Tampilkan pipeline lengkap:
    Sinyal asli | Spektrum FFT | Sinyal hasil IFFT
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Transformational Thinking — FFT Pipeline: {signal_name}",
                 fontsize=13, fontweight="bold")

    # Sinyal asli
    axes[0].plot(t, x_original, color=STYLE["continuous_color"], linewidth=1.8)
    axes[0].set_title("① Sinyal Asli  x(t)")
    axes[0].set_xlabel("Waktu")
    axes[0].set_ylabel("Amplitudo")
    _add_zero_line(axes[0])

    # Spektrum FFT
    axes[1].bar(freqs, magnitude[:len(freqs)],
                width=freqs[1] - freqs[0] if len(freqs) > 1 else 0.1,
                color=STYLE["freq_color"], alpha=0.85)
    axes[1].set_title("② Spektrum FFT  |X(f)|")
    axes[1].set_xlabel("Frekuensi (Hz)")
    axes[1].set_ylabel("Magnitude")

    # Rekonstruksi IFFT
    axes[2].plot(t, x_reconstructed[:len(t)],
                 color=STYLE["conv_color"], linewidth=1.8, linestyle="--")
    axes[2].set_title("③ Rekonstruksi IFFT  x̂(t)")
    axes[2].set_xlabel("Waktu")
    axes[2].set_ylabel("Amplitudo")
    _add_zero_line(axes[2])

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  6. PLOT KONVOLUSI (Input, Kernel, Output)
# ─────────────────────────────────────────────

def plot_convolution(x, h, y, method="Direct", signal_name=""):
    """
    Tampilkan sinyal input, kernel, dan hasil konvolusi.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Konvolusi ({method}) — {signal_name}",
                 fontsize=13, fontweight="bold")

    # Input
    axes[0].stem(range(len(x)), x,
                 linefmt=STYLE["continuous_color"],
                 markerfmt="o", basefmt="black")
    axes[0].set_title("① Sinyal Input  x[n]")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Amplitudo")
    _add_zero_line(axes[0])

    # Kernel
    axes[1].stem(range(len(h)), h,
                 linefmt=STYLE["kernel_color"],
                 markerfmt="s", basefmt="black")
    axes[1].set_title("② Kernel / Impuls Respons  h[n]")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Amplitudo")
    _add_zero_line(axes[1])

    # Hasil konvolusi
    axes[2].stem(range(len(y)), y,
                 linefmt=STYLE["conv_color"],
                 markerfmt="^", basefmt="black")
    axes[2].set_title("③ Hasil Konvolusi  y[n] = x[n] * h[n]")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("Amplitudo")
    _add_zero_line(axes[2])

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  7. PLOT PERBANDINGAN DIRECT vs FFT CONVOLUTION
# ─────────────────────────────────────────────

def plot_convolution_comparison(y_direct, y_fft):
    """Bandingkan hasil konvolusi langsung vs via FFT."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Perbandingan: Konvolusi Langsung vs via FFT",
                 fontsize=13, fontweight="bold")

    ax1.stem(range(len(y_direct)), y_direct,
             linefmt=STYLE["continuous_color"],
             markerfmt="o", basefmt="black")
    ax1.set_title("Konvolusi Langsung  O(N·M)")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Amplitudo")
    _add_zero_line(ax1)

    ax2.stem(range(len(y_fft)), y_fft,
             linefmt=STYLE["conv_color"],
             markerfmt="^", basefmt="black")
    ax2.set_title("Konvolusi via FFT  O(N log N)")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Amplitudo")
    _add_zero_line(ax2)

    plt.tight_layout()
    plt.show()
