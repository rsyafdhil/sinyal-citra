"""
audio_visualizer.py
===================
Fungsi visualisasi untuk sinyal audio 1D.
Konsisten dengan visualizer.py yang sudah ada.

Menggunakan matplotlib — tidak ada library audio eksternal.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# ── Konsisten dengan visualizer.py ──────────────────────
STYLE = {
    "waveform_color"   : "#2196F3",   # biru   — waveform
    "segment_color"    : "#E91E63",   # pink   — segmen terpilih
    "freq_color"       : "#9C27B0",   # ungu   — domain frekuensi
    "conv_color"       : "#4CAF50",   # hijau  — hasil konvolusi
    "kernel_color"     : "#FF9800",   # oranye — kernel/filter
    "echo_color"       : "#F44336",   # merah  — echo
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
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)


# ═══════════════════════════════════════════════════════
#  1. PLOT WAVEFORM AUDIO
# ═══════════════════════════════════════════════════════

def plot_waveform(t, samples, title="Waveform Audio",
                  start_sec=0.0, end_sec=None, show=True):
    """
    Plot waveform (domain waktu) sinyal audio.

    Parameter:
        t        : list waktu (detik)
        samples  : list float amplitude [-1.0, 1.0]
        title    : judul plot
        start_sec, end_sec : label segmen (untuk judul)
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax.plot(t, samples, color=STYLE["waveform_color"],
            linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    ax.set_ylim(-1.1, 1.1)
    _add_zero_line(ax)

    if end_sec is not None:
        ax.set_title(
            f"Segmen: {start_sec:.2f}s – {end_sec:.2f}s  "
            f"| {len(samples):,} sampel",
            fontsize=10, color="gray"
        )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════
#  2. PLOT SPEKTRUM FFT AUDIO
# ═══════════════════════════════════════════════════════

def plot_spectrum(freqs, magnitude, title="Spektrum Frekuensi Audio",
                  log_scale=True, show=True):
    """
    Plot spektrum frekuensi hasil FFT sinyal audio.

    Parameter:
        freqs     : list frekuensi (Hz) — sisi positif saja
        magnitude : list magnitude (|X[k]|)
        log_scale : tampilkan magnitude dalam skala dB
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    if log_scale:
        # Konversi ke dB
        mag_db = [20 * math.log10(m + 1e-10) for m in magnitude]
        ax.plot(freqs, mag_db, color=STYLE["freq_color"], linewidth=0.9)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Skala logaritmik (dB) — lebih jelas untuk sinyal audio",
                     fontsize=9, color="gray")
    else:
        ax.plot(freqs, magnitude, color=STYLE["freq_color"], linewidth=0.9)
        ax.set_ylabel("Magnitude")

    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_xlim(0, max(freqs) if freqs else 1)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════
#  3. PLOT DUAL DOMAIN AUDIO (Waveform + Spektrum)
# ═══════════════════════════════════════════════════════

def plot_audio_dual_domain(t, samples, freqs, magnitude,
                            sample_rate, start_sec=0.0,
                            signal_name="Audio Segment"):
    """
    Tampilkan waveform (kiri) dan spektrum frekuensi (kanan) secara berdampingan.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(f"Dual Domain — {signal_name}", fontsize=13, fontweight="bold")

    # ── Waveform ──────────────────────────────────────
    ax1.plot(t, samples, color=STYLE["waveform_color"], linewidth=0.7, alpha=0.85)
    ax1.set_title(f"Domain Waktu  x(t)  [{start_sec:.1f}s – {start_sec + len(samples)/sample_rate:.1f}s]",
                  fontsize=10)
    ax1.set_xlabel("Waktu (s)")
    ax1.set_ylabel("Amplitudo")
    ax1.set_ylim(-1.1, 1.1)
    _add_zero_line(ax1)

    # ── Spektrum ─────────────────────────────────────
    mag_db = [20 * math.log10(m + 1e-10) for m in magnitude]
    ax2.plot(freqs, mag_db, color=STYLE["freq_color"], linewidth=0.9)
    ax2.set_title("Domain Frekuensi  |X(f)|  (dB)", fontsize=10)
    ax2.set_xlabel("Frekuensi (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(0, sample_rate / 2)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
#  4. PLOT KONVOLUSI AUDIO (Original + Filter + Output)
# ═══════════════════════════════════════════════════════

def plot_audio_convolution(t_orig, x_orig,
                            t_out, y_out,
                            filter_name="Moving Average",
                            signal_name="Audio"):
    """
    Tampilkan sinyal asli dan hasil konvolusi (filter/efek).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)
    fig.suptitle(f"Konvolusi Audio — {filter_name} | {signal_name}",
                 fontsize=13, fontweight="bold")

    # Sinyal asli
    ax1.plot(t_orig, x_orig, color=STYLE["waveform_color"],
             linewidth=0.7, alpha=0.9, label="Input  x[n]")
    ax1.set_title("① Sinyal Input (Original)", fontsize=10)
    ax1.set_xlabel("Waktu (s)")
    ax1.set_ylabel("Amplitudo")
    ax1.set_ylim(-1.1, 1.1)
    _add_zero_line(ax1)
    ax1.legend(loc="upper right", fontsize=8)

    # Output konvolusi
    # Buat sumbu waktu untuk output (bisa lebih panjang karena konvolusi)
    t_out_plot = t_out[:len(y_out)]
    ax2.plot(t_out_plot, y_out[:len(t_out_plot)],
             color=STYLE["conv_color"], linewidth=0.7, alpha=0.9,
             label=f"Output  y[n] = x[n] * h[n]")
    ax2.set_title(f"② Hasil Filter: {filter_name}", fontsize=10)
    ax2.set_xlabel("Waktu (s)")
    ax2.set_ylabel("Amplitudo")
    _add_zero_line(ax2)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
#  5. PLOT PERBANDINGAN SEBELUM vs SESUDAH FILTER
# ═══════════════════════════════════════════════════════

def plot_filter_comparison(t, x_orig, y_filtered,
                            freqs_orig, mag_orig,
                            freqs_filt, mag_filt,
                            filter_name="Low-Pass Filter"):
    """
    Pipeline lengkap: Waveform asli | Waveform filtered | Spektrum perbandingan.
    """
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Analisis Filter Audio — {filter_name}",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])   # span dua kolom

    # ── Waveform asli ─────────────────────────────────
    ax1.plot(t, x_orig, color=STYLE["waveform_color"], linewidth=0.7)
    ax1.set_title("① Waveform Asli  x[n]", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Waktu (s)")
    ax1.set_ylabel("Amplitudo")
    ax1.set_ylim(-1.1, 1.1)
    _add_zero_line(ax1)

    # ── Waveform setelah filter ───────────────────────
    t_filt = t[:len(y_filtered)]
    ax2.plot(t_filt, y_filtered[:len(t_filt)],
             color=STYLE["conv_color"], linewidth=0.7)
    ax2.set_title(f"② Waveform Setelah {filter_name}  y[n]",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Waktu (s)")
    ax2.set_ylabel("Amplitudo")
    _add_zero_line(ax2)

    # ── Spektrum perbandingan ─────────────────────────
    mag_db_orig = [20 * math.log10(m + 1e-10) for m in mag_orig]
    mag_db_filt = [20 * math.log10(m + 1e-10) for m in mag_filt]

    ax3.plot(freqs_orig, mag_db_orig, color=STYLE["waveform_color"],
             linewidth=1.0, alpha=0.8, label="Spektrum Asli")
    ax3.plot(freqs_filt, mag_db_filt, color=STYLE["conv_color"],
             linewidth=1.2, alpha=0.9, label=f"Setelah {filter_name}", linestyle="--")
    ax3.set_title("③ Perbandingan Spektrum Frekuensi (dB)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Frekuensi (Hz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
#  6. PLOT ECHO EFFECT
# ═══════════════════════════════════════════════════════

def plot_echo_effect(t_orig, x_orig, t_echo, y_echo,
                     delay_sec, decay, signal_name="Audio"):
    """
    Visualisasi efek echo: sinyal asli vs sinyal dengan echo.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.suptitle(
        f"Echo Effect — delay={delay_sec:.2f}s, decay={decay:.2f} | {signal_name}",
        fontsize=13, fontweight="bold"
    )

    # Sinyal asli
    axes[0].plot(t_orig, x_orig, color=STYLE["waveform_color"],
                 linewidth=0.8, label="Original")
    axes[0].set_title("① Sinyal Asli  x[n]", fontsize=10)
    axes[0].set_xlabel("Waktu (s)")
    axes[0].set_ylabel("Amplitudo")
    axes[0].set_ylim(-1.1, 1.1)
    _add_zero_line(axes[0])
    axes[0].legend(fontsize=8)

    # Sinyal dengan echo
    t_echo_plot = t_echo[:len(y_echo)]
    axes[1].plot(t_echo_plot, y_echo[:len(t_echo_plot)],
                 color=STYLE["echo_color"], linewidth=0.8, alpha=0.85,
                 label=f"Echo (delay={delay_sec:.2f}s, decay={decay:.2f})")
    axes[1].set_title("② Sinyal dengan Echo  y[n]", fontsize=10)
    axes[1].set_xlabel("Waktu (s)")
    axes[1].set_ylabel("Amplitudo")
    _add_zero_line(axes[1])
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
#  7. PLOT INFO RINGKAS (statistik sinyal)
# ═══════════════════════════════════════════════════════

def print_audio_info(info, stats, start_sec, end_sec, window_name="Rectangular"):
    """Cetak ringkasan info audio ke terminal."""
    print("\n" + "─" * 50)
    print("  INFO FILE WAV")
    print("─" * 50)
    print(info)
    print("─" * 50)
    print("  INFO SEGMEN")
    print("─" * 50)
    print(f"  Segmen        : {start_sec:.2f}s – {end_sec:.2f}s")
    print(f"  Durasi segmen : {end_sec - start_sec:.2f} detik")
    print(f"  Jumlah sampel : {stats['n_samples']:,}")
    print(f"  Windowing     : {window_name}")
    print(f"  RMS Amplitude : {stats['rms']:.4f}")
    print(f"  Peak          : {stats['peak']:.4f} ({stats['peak_db']:.1f} dBFS)")
    print("─" * 50)
