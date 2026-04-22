"""
main.py
=======
Program utama — Menu interaktif demo pengolahan sinyal.
Mencakup semua topik sesuai silabus:
  1. Mathematical Language of Signals (Sinyal Dasar)
  2. What is a Signal? (Kontinu vs Diskrit)
  3. The Dual Domains (Waktu vs Frekuensi)
  4. Transformational Thinking (FFT Pipeline)
  5. Systems and Convolution
  ── Pengolahan Sinyal 2D (Gambar) ──
  6. Sinyal 2D & Gambar Dasar
  7. Konvolusi 2D (Spatial Domain)
  8. FFT 2D & Filtering Frekuensi
  ── Pengolahan Sinyal Audio ──────────
  10. Audio Signal Processing (WAV)
"""

from signals import (
    delta_signal, step_signal,
    sine_signal, cosine_signal,
    exponential_signal, discrete_sine,
    composite_signal
)
from transforms import (
    dft, fft, ifft, idft,
    get_magnitude, get_frequency_axis
)
from convolution import (
    convolve_direct, convolve_fft,
    kernel_moving_average, kernel_impulse,
    kernel_derivative, kernel_gaussian
)
from visualizer import (
    plot_signal, plot_all_basic_signals,
    plot_continuous_vs_discrete,
    plot_dual_domain, plot_fft_pipeline,
    plot_convolution, plot_convolution_comparison
)

# ── Modul 2D ──
from menu_2d import menu_sinyal_2d, menu_konvolusi_2d, menu_fft2d

# ── Modul Audio (BARU) ──
from menu_audio import menu_audio


# ═══════════════════════════════════════════════════════
#  HELPER UI
# ═══════════════════════════════════════════════════════

def clear():
    print("\n" * 2)


def header(title):
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)


def sub_header(title):
    print("\n" + "-" * 45)
    print(f"  {title}")
    print("-" * 45)


def pause():
    input("\n  [Enter] untuk kembali ke menu...")


# ═══════════════════════════════════════════════════════
#  MODUL 1 — SINYAL DASAR
# ═══════════════════════════════════════════════════════

def menu_sinyal_dasar():
    while True:
        clear()
        header("1. SINYAL DASAR (Mathematical Language of Signals)")
        print("  [1] Delta / Impulse Signal  δ[n]")
        print("  [2] Step Signal             u[n]")
        print("  [3] Sinyal Sinus            sin(t)")
        print("  [4] Sinyal Cosinus          cos(t)")
        print("  [5] Sinyal Eksponensial     e^(-αt)")
        print("  [6] Tampilkan Semua Sinyal Dasar")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            sub_header("Delta / Impulse Signal  δ[n]")
            n, x = delta_signal(n_range=(-10, 10), shift=0)
            plot_signal(n, x, title="Delta Signal  δ[n]",
                        xlabel="n", discrete=True)
            pause()

        elif pilihan == "2":
            sub_header("Step Signal  u[n]")
            n, x = step_signal(n_range=(-10, 10), shift=0)
            plot_signal(n, x, title="Step Signal  u[n]",
                        xlabel="n", discrete=True)
            pause()

        elif pilihan == "3":
            sub_header("Sinyal Sinus  x(t) = sin(2πft)")
            t, x = sine_signal(frequency=2.0, amplitude=1.0, t_range=(0, 2))
            plot_signal(t, x, title="Sinyal Sinus  x(t) = sin(2π·2·t)",
                        xlabel="Waktu (s)")
            pause()

        elif pilihan == "4":
            sub_header("Sinyal Cosinus  x(t) = cos(2πft)")
            t, x = cosine_signal(frequency=2.0, amplitude=1.0, t_range=(0, 2))
            plot_signal(t, x, title="Sinyal Cosinus  x(t) = cos(2π·2·t)",
                        xlabel="Waktu (s)", color="#FF5722")
            pause()

        elif pilihan == "5":
            sub_header("Sinyal Eksponensial  x(t) = e^(-αt)")
            t, x = exponential_signal(alpha=1.0, t_range=(0, 5))
            plot_signal(t, x, title="Sinyal Eksponensial  x(t) = e^(-t)",
                        xlabel="Waktu (s)", color="#009688")
            pause()

        elif pilihan == "6":
            sub_header("Semua Sinyal Dasar")
            n_d, x_d   = delta_signal()
            n_s, x_s   = step_signal()
            t_sin, x_sin = sine_signal(frequency=2.0)
            t_cos, x_cos = cosine_signal(frequency=2.0)
            t_exp, x_exp = exponential_signal(alpha=1.0)
            n_ds, x_ds  = discrete_sine(frequency=2.0, sampling_rate=20)

            signals_data = [
                {"title": "Delta Signal  δ[n]",        "t": n_d,   "x": x_d,   "discrete": True},
                {"title": "Step Signal  u[n]",          "t": n_s,   "x": x_s,   "discrete": True},
                {"title": "Sinus  sin(2π·2·t)",         "t": t_sin, "x": x_sin, "discrete": False},
                {"title": "Cosinus  cos(2π·2·t)",       "t": t_cos, "x": x_cos, "discrete": False},
                {"title": "Eksponensial  e^(-t)",       "t": t_exp, "x": x_exp, "discrete": False},
                {"title": "Diskrit Sinus  x[n]",        "t": n_ds,  "x": x_ds,  "discrete": True},
            ]
            plot_all_basic_signals(signals_data)
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 2 — KONTINU vs DISKRIT
# ═══════════════════════════════════════════════════════

def menu_kontinu_diskrit():
    while True:
        clear()
        header("2. WHAT IS A SIGNAL? — Kontinu vs Diskrit")
        print("  [1] Sinus: Kontinu vs Diskrit (fs=10 Hz)")
        print("  [2] Sinus: Kontinu vs Diskrit (fs=40 Hz)")
        print("  [3] Eksponensial: Kontinu vs Diskrit")
        print("  [4] Tampilkan Semua Perbandingan")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            t_c, x_c = sine_signal(frequency=2.0, t_range=(0, 2), num_points=500)
            n_d, x_d = discrete_sine(frequency=2.0, sampling_rate=10, t_range=(0, 2))
            t_d_plot  = [i / 10 for i in n_d]
            plot_continuous_vs_discrete(t_c, x_c, t_d_plot, x_d,
                                         "Sinus 2 Hz  (fs=10 Hz — undersampling)")
            pause()

        elif pilihan == "2":
            t_c, x_c = sine_signal(frequency=2.0, t_range=(0, 2), num_points=500)
            n_d, x_d = discrete_sine(frequency=2.0, sampling_rate=40, t_range=(0, 2))
            t_d_plot  = [i / 40 for i in n_d]
            plot_continuous_vs_discrete(t_c, x_c, t_d_plot, x_d,
                                         "Sinus 2 Hz  (fs=40 Hz — oversampling)")
            pause()

        elif pilihan == "3":
            t_c, x_c = exponential_signal(alpha=1.0, t_range=(0, 5), num_points=500)
            import math
            fs = 10
            n_d = list(range(int(5 * fs)))
            x_d = [math.exp(-i / fs) for i in n_d]
            t_d_plot = [i / fs for i in n_d]
            plot_continuous_vs_discrete(t_c, x_c, t_d_plot, x_d,
                                         "Eksponensial  e^(-t)  (fs=10 Hz)")
            pause()

        elif pilihan == "4":
            for label, fs in [("Sinus (fs=10 Hz)", 10), ("Sinus (fs=40 Hz)", 40)]:
                t_c, x_c = sine_signal(frequency=2.0, t_range=(0, 2), num_points=500)
                n_d, x_d = discrete_sine(frequency=2.0, sampling_rate=fs, t_range=(0, 2))
                t_d_plot = [i / fs for i in n_d]
                plot_continuous_vs_discrete(t_c, x_c, t_d_plot, x_d, label)
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 3 — THE DUAL DOMAINS
# ═══════════════════════════════════════════════════════

def menu_dual_domain():
    while True:
        clear()
        header("3. THE DUAL DOMAINS — Waktu vs Frekuensi")
        print("  [1] Sinus Tunggal (1 frekuensi)")
        print("  [2] Sinyal Gabungan (multi-frekuensi)")
        print("  [3] Tampilkan Keduanya")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            fs = 100
            t, x = sine_signal(frequency=5.0, t_range=(0, 1), num_points=fs)
            X    = fft(x)
            mag  = get_magnitude(X)
            freqs = get_frequency_axis(len(X), sampling_rate=fs)
            plot_dual_domain(t, x, freqs, mag, "Sinus 5 Hz")
            pause()

        elif pilihan == "2":
            fs = 200
            t, x = composite_signal(
                frequencies=[3.0, 7.0, 15.0],
                amplitudes=[1.0, 0.6, 0.4],
                t_range=(0, 1), num_points=fs
            )
            X     = fft(x)
            mag   = get_magnitude(X)
            freqs = get_frequency_axis(len(X), sampling_rate=fs)
            plot_dual_domain(t, x, freqs, mag,
                             "Sinyal Gabungan (3 Hz + 7 Hz + 15 Hz)")
            pause()

        elif pilihan == "3":
            fs = 100
            t1, x1 = sine_signal(frequency=5.0, t_range=(0, 1), num_points=fs)
            X1 = fft(x1)
            mag1 = get_magnitude(X1)
            freqs1 = get_frequency_axis(len(X1), fs)
            plot_dual_domain(t1, x1, freqs1, mag1, "Sinus 5 Hz")

            fs2 = 200
            t2, x2 = composite_signal([3.0, 7.0, 15.0], [1.0, 0.6, 0.4],
                                       t_range=(0, 1), num_points=fs2)
            X2 = fft(x2)
            mag2 = get_magnitude(X2)
            freqs2 = get_frequency_axis(len(X2), fs2)
            plot_dual_domain(t2, x2, freqs2, mag2, "Sinyal Gabungan (3+7+15 Hz)")
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 4 — TRANSFORMATIONAL THINKING (FFT PIPELINE)
# ═══════════════════════════════════════════════════════

def menu_fft_pipeline():
    while True:
        clear()
        header("4. TRANSFORMATIONAL THINKING — FFT Pipeline")
        print("  [1] Pipeline: Sinus Tunggal")
        print("  [2] Pipeline: Sinyal Gabungan (multi-frekuensi)")
        print("  [3] Tampilkan Keduanya")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan in ("1", "3"):
            fs = 128
            t, x = sine_signal(frequency=5.0, t_range=(0, 1), num_points=fs)
            X    = fft(x)
            mag  = get_magnitude(X)
            freqs = get_frequency_axis(len(X), sampling_rate=fs)
            x_rec = ifft(X)
            plot_fft_pipeline(t, x, freqs, mag, x_rec, "Sinus 5 Hz")
            if pilihan == "1":
                pause()

        if pilihan in ("2", "3"):
            fs = 256
            t, x = composite_signal(
                [3.0, 7.0, 15.0], [1.0, 0.6, 0.4],
                t_range=(0, 1), num_points=fs
            )
            X     = fft(x)
            mag   = get_magnitude(X)
            freqs = get_frequency_axis(len(X), sampling_rate=fs)
            x_rec = ifft(X)
            plot_fft_pipeline(t, x, freqs, mag, x_rec,
                              "Sinyal Gabungan (3+7+15 Hz)")
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 5 — SYSTEMS AND CONVOLUTION
# ═══════════════════════════════════════════════════════

def menu_konvolusi():
    while True:
        clear()
        header("5. SYSTEMS AND CONVOLUTION")
        print("  [1] Konvolusi Langsung — Delta * Moving Average")
        print("  [2] Konvolusi Langsung — Sinus * Gaussian")
        print("  [3] Konvolusi via FFT  — Sinus * Moving Average")
        print("  [4] Perbandingan Direct vs FFT Convolution")
        print("  [5] Tampilkan Semua Demo Konvolusi")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan in ("1", "5"):
            n, x = delta_signal(n_range=(0, 30), shift=5)
            h    = kernel_moving_average(size=5)
            y    = convolve_direct(x, h)
            plot_convolution(x, h, y,
                             method="Langsung",
                             signal_name="δ[n-5] * Moving Average")
            if pilihan == "1":
                pause()

        if pilihan in ("2", "5"):
            _, x_sin = discrete_sine(frequency=1.0, sampling_rate=30,
                                      t_range=(0, 2))
            h_gauss  = kernel_gaussian(size=11, sigma=2.0)
            y        = convolve_direct(x_sin, h_gauss)
            plot_convolution(x_sin, h_gauss, y,
                             method="Langsung",
                             signal_name="Sinus Diskrit * Gaussian")
            if pilihan == "2":
                pause()

        if pilihan in ("3", "5"):
            _, x_sin = discrete_sine(frequency=1.0, sampling_rate=30,
                                      t_range=(0, 2))
            h_ma = kernel_moving_average(size=7)
            y    = convolve_fft(x_sin, h_ma)
            plot_convolution(x_sin, h_ma, y,
                             method="via FFT",
                             signal_name="Sinus Diskrit * Moving Average")
            if pilihan == "3":
                pause()

        if pilihan in ("4", "5"):
            _, x_sin = discrete_sine(frequency=1.0, sampling_rate=30,
                                      t_range=(0, 2))
            h_ma    = kernel_moving_average(size=7)
            y_dir   = convolve_direct(x_sin, h_ma)
            y_fft   = convolve_fft(x_sin, h_ma)
            plot_convolution_comparison(y_dir, y_fft)
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 9 — TAMPILKAN SEMUA GRAFIK
# ═══════════════════════════════════════════════════════

def tampilkan_semua():
    print("\n  Menampilkan semua grafik 1D... (tunggu sebentar)\n")

    n_d, x_d     = delta_signal()
    n_s, x_s     = step_signal()
    t_sin, x_sin = sine_signal(frequency=2.0)
    t_cos, x_cos = cosine_signal(frequency=2.0)
    t_exp, x_exp = exponential_signal(alpha=1.0)
    n_ds, x_ds   = discrete_sine(frequency=2.0, sampling_rate=20)

    plot_all_basic_signals([
        {"title": "Delta Signal  δ[n]",  "t": n_d,   "x": x_d,   "discrete": True},
        {"title": "Step Signal  u[n]",   "t": n_s,   "x": x_s,   "discrete": True},
        {"title": "Sinus Kontinu",       "t": t_sin, "x": x_sin, "discrete": False},
        {"title": "Cosinus Kontinu",     "t": t_cos, "x": x_cos, "discrete": False},
        {"title": "Eksponensial",        "t": t_exp, "x": x_exp, "discrete": False},
        {"title": "Sinus Diskrit x[n]",  "t": n_ds,  "x": x_ds,  "discrete": True},
    ])

    t_c, x_c = sine_signal(frequency=2.0, t_range=(0, 2), num_points=500)
    n_d2, x_d2 = discrete_sine(frequency=2.0, sampling_rate=20, t_range=(0, 2))
    t_d_plot = [i / 20 for i in n_d2]
    plot_continuous_vs_discrete(t_c, x_c, t_d_plot, x_d2, "Sinus 2 Hz (fs=20 Hz)")

    fs = 200
    t, x = composite_signal([3.0, 7.0, 15.0], [1.0, 0.6, 0.4],
                             t_range=(0, 1), num_points=fs)
    X     = fft(x)
    mag   = get_magnitude(X)
    freqs = get_frequency_axis(len(X), fs)
    plot_dual_domain(t, x, freqs, mag, "Sinyal Gabungan (3+7+15 Hz)")

    fs = 256
    t, x = composite_signal([3.0, 7.0, 15.0], [1.0, 0.6, 0.4],
                             t_range=(0, 1), num_points=fs)
    X     = fft(x)
    mag   = get_magnitude(X)
    freqs = get_frequency_axis(len(X), fs)
    x_rec = ifft(X)
    plot_fft_pipeline(t, x, freqs, mag, x_rec, "Sinyal Gabungan")

    _, x_sin = discrete_sine(frequency=1.0, sampling_rate=30, t_range=(0, 2))
    h_ma     = kernel_moving_average(size=7)
    y_dir    = convolve_direct(x_sin, h_ma)
    y_fft_c  = convolve_fft(x_sin, h_ma)
    plot_convolution(x_sin, h_ma, y_dir, method="Langsung",
                     signal_name="Sinus * Moving Average")
    plot_convolution_comparison(y_dir, y_fft_c)

    pause()


# ═══════════════════════════════════════════════════════
#  MENU UTAMA
# ═══════════════════════════════════════════════════════

def main():
    while True:
        clear()
        print("╔═══════════════════════════════════════════════════════╗")
        print("║      SIGNAL PROCESSING DEMO — Python Edition          ║")
        print("║       Pengolahan Sinyal, Sistem, Citra & Audio         ║")
        print("╠═══════════════════════════════════════════════════════╣")
        print("║  ── Sinyal 1D ────────────────────────────────────── ║")
        print("║  [1]  Sinyal Dasar (Mathematical Language)             ║")
        print("║  [2]  What is a Signal? (Kontinu vs Diskrit)           ║")
        print("║  [3]  The Dual Domains (Waktu vs Frekuensi)            ║")
        print("║  [4]  Transformational Thinking (FFT Pipeline)         ║")
        print("║  [5]  Systems and Convolution                          ║")
        print("║  ── Sinyal 2D (Gambar / Citra) ─────────────────────  ║")
        print("║  [6]  Sinyal 2D & Gambar Dasar                         ║")
        print("║  [7]  Konvolusi 2D (Spatial Domain)                    ║")
        print("║  [8]  FFT 2D & Filtering Frekuensi                     ║")
        print("║  ── Sinyal Audio ────────────────────────────────────  ║")
        print("║  [10] Audio Signal Processing (File WAV)               ║")
        print("║  ─────────────────────────────────────────────────── ║")
        print("║  [9]  Tampilkan Semua Grafik 1D                        ║")
        print("║  [0]  Keluar                                            ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print()
        pilihan = input("  Pilih menu: ").strip()

        if   pilihan == "1":  menu_sinyal_dasar()
        elif pilihan == "2":  menu_kontinu_diskrit()
        elif pilihan == "3":  menu_dual_domain()
        elif pilihan == "4":  menu_fft_pipeline()
        elif pilihan == "5":  menu_konvolusi()
        elif pilihan == "6":  menu_sinyal_2d()
        elif pilihan == "7":  menu_konvolusi_2d()
        elif pilihan == "8":  menu_fft2d()
        elif pilihan == "10": menu_audio()
        elif pilihan == "9":  tampilkan_semua()
        elif pilihan == "0":
            print("\n  Sampai jumpa! 👋\n")
            break
        else:
            print("  Input tidak valid, coba lagi.")
            input("  [Enter] untuk lanjut...")


if __name__ == "__main__":
    main()
