"""
menu_audio.py
=============
Menu interaktif untuk analisis sinyal audio 1D.
Terintegrasi dengan:
  - transforms.py     (FFT/IFFT manual)
  - convolution.py    (konvolusi langsung & via FFT)
  - audio_signals.py  (WAV reader dari scratch)
  - audio_filters.py  (kernel filter audio)
  - audio_visualizer.py (plotting)

Topik:
  10a. Load & Eksplorasi File WAV
  10b. Spektrum FFT Audio
  10c. Konvolusi Audio — Filter (Low-Pass, High-Pass)
  10d. Efek Echo & Reverb
"""

import os

from audio_signals import (
    read_wav, get_segment, get_time_axis,
    downsample, apply_hann_window,
    apply_hamming_window, apply_rectangular_window,
    signal_stats
)
from audio_filters import (
    kernel_audio_lowpass_moving_average,
    kernel_audio_lowpass_gaussian,
    kernel_audio_highpass_simple,
    kernel_audio_highpass_from_lowpass,
    kernel_audio_trebleboost,
    kernel_audio_bassboost,
    kernel_audio_echo,
    kernel_audio_reverb,
    normalize_audio,
    trim_to_original_length
)
from audio_visualizer import (
    plot_waveform, plot_spectrum,
    plot_audio_dual_domain,
    plot_audio_convolution,
    plot_filter_comparison,
    plot_echo_effect,
    print_audio_info
)

# Pakai FFT & konvolusi yang sudah ada!
from transforms import fft, ifft, get_magnitude, get_frequency_axis
from convolution import convolve_fft, convolve_direct


# ═══════════════════════════════════════════════════════
#  HELPER UI
# ═══════════════════════════════════════════════════════

def clear():
    print("\n" * 2)

def header(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def sub_header(title):
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)

def pause():
    input("\n  [Enter] untuk kembali ke menu...")


# ═══════════════════════════════════════════════════════
#  HELPER: INPUT PENGGUNA
# ═══════════════════════════════════════════════════════

def _input_filepath():
    """Minta path file WAV dari pengguna."""
    while True:
        filepath = input("\n  Masukkan path file WAV: ").strip()
        if not filepath:
            print("  ⚠ Path tidak boleh kosong.")
            continue
        if not os.path.exists(filepath):
            print(f"  ⚠ File tidak ditemukan: '{filepath}'")
            retry = input("  Coba lagi? [y/n]: ").strip().lower()
            if retry != 'y':
                return None
            continue
        if not filepath.lower().endswith('.wav'):
            print("  ⚠ File harus berformat .wav")
            retry = input("  Tetap lanjutkan? [y/n]: ").strip().lower()
            if retry != 'y':
                continue
        return filepath


def _input_segment(info):
    """Minta input start/end detik dari pengguna."""
    print(f"\n  Durasi file: {info.duration_sec:.2f} detik "
          f"({info.duration_sec/60:.2f} menit)")
    print(f"  Contoh input: 10.5 untuk 10 detik 500ms")

    while True:
        try:
            start_str = input(f"  Mulai dari detik ke (default 0): ").strip()
            start_sec = float(start_str) if start_str else 0.0
        except ValueError:
            print("  Input tidak valid, masukkan angka.")
            continue

        while True:
            try:
                end_str = input(
                    f"  Sampai detik ke (default {min(start_sec+5.0, info.duration_sec):.1f},"
                    f" max {info.duration_sec:.2f}): "
                ).strip()
                if end_str:
                    end_sec = float(end_str)
                else:
                    end_sec = min(start_sec + 5.0, info.duration_sec)
            except ValueError:
                print("  Input tidak valid, masukkan angka.")
                continue
            break

        if start_sec >= end_sec:
            print("  ⚠ Detik awal harus lebih kecil dari detik akhir.")
            continue
        if start_sec < 0 or end_sec > info.duration_sec:
            print(f"  ⚠ Rentang harus antara 0 dan {info.duration_sec:.2f}.")
            continue

        return start_sec, end_sec


def _input_window():
    """Pilih windowing function."""
    print("\n  Pilih windowing (untuk FFT):")
    print("  [1] Rectangular (tidak ada windowing)")
    print("  [2] Hann  — recommended untuk musik")
    print("  [3] Hamming")
    pilihan = input("  Pilih [1-3, default=2]: ").strip()
    if pilihan == "1":
        return apply_rectangular_window, "Rectangular"
    elif pilihan == "3":
        return apply_hamming_window, "Hamming"
    else:
        return apply_hann_window, "Hann"


def _load_wav_interactive():
    """
    Helper: load WAV + pilih segmen.
    Return: (segment, info, t, start_sec, end_sec) atau None jika gagal.
    """
    filepath = _input_filepath()
    if filepath is None:
        return None

    print("\n  Membaca file WAV...")
    try:
        samples, info = read_wav(filepath)
    except Exception as e:
        print(f"\n  ✗ Gagal membaca file: {e}")
        return None

    print(f"\n  ✓ File berhasil dibaca!")
    print(info)

    start_sec, end_sec = _input_segment(info)

    segment, actual_start, actual_end = get_segment(
        samples, info, start_sec, end_sec
    )
    t = get_time_axis(segment, info.sample_rate, actual_start)

    print(f"\n  ✓ Segmen diambil: {actual_start:.2f}s – {actual_end:.2f}s")
    print(f"    Jumlah sampel: {len(segment):,}")

    return segment, info, t, actual_start, actual_end


def _prepare_fft(segment, info, window_fn, window_name, max_samples=8192):
    """
    Siapkan segmen untuk FFT:
      1. Downsample jika terlalu panjang
      2. Terapkan windowing
      3. Hitung FFT

    Return: (t_ds, seg_ds, freqs, magnitude, sr_used)
    """
    seg = list(segment)
    sr  = info.sample_rate

    # Downsample jika sampel terlalu banyak (FFT masih cukup cepat)
    if len(seg) > max_samples:
        target_sr = int(sr * max_samples / len(seg))
        target_sr = max(8000, target_sr)  # minimal 8kHz
        seg, sr = downsample(seg, sr, target_sr)
        print(f"  (Downsampled ke {sr} Hz untuk kecepatan FFT)")

    # Windowing
    seg_windowed = window_fn(seg)

    # FFT manual dari transforms.py
    X     = fft(seg_windowed)
    mag   = get_magnitude(X)
    freqs = get_frequency_axis(len(X), sampling_rate=sr)

    # Sisi positif saja (Nyquist)
    half  = len(X) // 2
    mag   = mag[:half]
    freqs = freqs[:half]

    return seg, freqs, mag, sr


# ═══════════════════════════════════════════════════════
#  MENU 10A — LOAD & EKSPLORASI
# ═══════════════════════════════════════════════════════

def menu_load_eksplorasi():
    clear()
    header("10A. LOAD & EKSPLORASI FILE WAV")

    result = _load_wav_interactive()
    if result is None:
        pause()
        return

    segment, info, t, start_sec, end_sec = result
    stats = signal_stats(segment)

    # Info ringkasan
    print_audio_info(info, stats, start_sec, end_sec, "Rectangular")

    # Plot waveform
    print("\n  Menampilkan waveform...")
    plot_waveform(t, segment,
                  title=f"Waveform Audio",
                  start_sec=start_sec, end_sec=end_sec)
    pause()


# ═══════════════════════════════════════════════════════
#  MENU 10B — SPEKTRUM FFT
# ═══════════════════════════════════════════════════════

def menu_spektrum_fft():
    clear()
    header("10B. SPEKTRUM FFT AUDIO")
    print("  Menggunakan FFT manual dari transforms.py\n")

    result = _load_wav_interactive()
    if result is None:
        pause()
        return

    segment, info, t, start_sec, end_sec = result
    window_fn, window_name = _input_window()

    print("\n  Menghitung FFT...")
    seg_ds, freqs, mag, sr_used = _prepare_fft(segment, info, window_fn, window_name)
    t_ds = get_time_axis(seg_ds, sr_used, start_sec)

    stats = signal_stats(seg_ds)
    print_audio_info(info, stats, start_sec, end_sec, window_name)

    # Pilih tampilan
    print("\n  Pilih tampilan:")
    print("  [1] Spektrum saja")
    print("  [2] Dual domain (Waveform + Spektrum)")
    view = input("  Pilih [1/2, default=2]: ").strip()

    if view == "1":
        plot_spectrum(freqs, mag,
                      title=f"Spektrum FFT — Segmen {start_sec:.1f}s–{end_sec:.1f}s "
                            f"(Window: {window_name})")
    else:
        plot_audio_dual_domain(
            t_ds, seg_ds, freqs, mag,
            sample_rate=sr_used,
            start_sec=start_sec,
            signal_name=f"Segmen {start_sec:.1f}s–{end_sec:.1f}s | Window: {window_name}"
        )

    # Deteksi frekuensi dominan
    print("\n  ── Top 5 Frekuensi Dominan ─────────────────")
    indexed = sorted(enumerate(mag), key=lambda x: x[1], reverse=True)
    shown = []
    count = 0
    for idx, m in indexed:
        if count >= 5:
            break
        f = freqs[idx]
        # Skip frekuensi yang terlalu dekat satu sama lain (min 20 Hz jarak)
        if all(abs(f - sf) > 20 for sf in shown):
            import math
            db = 20 * math.log10(m + 1e-10)
            print(f"    {f:7.1f} Hz  →  {db:6.1f} dB")
            shown.append(f)
            count += 1

    pause()


# ═══════════════════════════════════════════════════════
#  MENU 10C — FILTER AUDIO (KONVOLUSI)
# ═══════════════════════════════════════════════════════

def menu_filter_audio():
    while True:
        clear()
        header("10C. FILTER AUDIO — KONVOLUSI 1D")
        print("  Menggunakan convolve_fft() dari convolution.py\n")
        print("  [1] Low-Pass Filter (Moving Average) — meredam treble")
        print("  [2] Low-Pass Filter (Gaussian)       — lebih halus")
        print("  [3] Bass Boost                        — perkuat bass")
        print("  [4] High-Pass Filter                  — meredam bass")
        print("  [5] Treble Boost                      — perkuat treble")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "0":
            break

        if pilihan not in ("1","2","3","4","5"):
            print("  Input tidak valid.")
            input("  [Enter] lanjut...")
            continue

        result = _load_wav_interactive()
        if result is None:
            pause()
            continue

        segment, info, t, start_sec, end_sec = result
        window_fn, window_name = _input_window()

        # Downsample kalau perlu (untuk kecepatan konvolusi FFT)
        seg = list(segment)
        sr  = info.sample_rate
        if len(seg) > 16000:
            seg, sr = downsample(seg, sr, 16000)
            print(f"  (Downsampled ke {sr} Hz)")
        t_seg = get_time_axis(seg, sr, start_sec)

        # Pilih kernel sesuai menu
        if pilihan == "1":
            size = 31
            try:
                s = input(f"  Ukuran moving average [default {size}]: ").strip()
                if s: size = int(s)
            except ValueError:
                pass
            h = kernel_audio_lowpass_moving_average(size=size)
            filter_name = f"Low-Pass Moving Average (size={size})"

        elif pilihan == "2":
            size, sigma = 51, 10.0
            try:
                s = input(f"  Ukuran kernel Gaussian [default {size}]: ").strip()
                if s: size = int(s)
                s = input(f"  Sigma [default {sigma}]: ").strip()
                if s: sigma = float(s)
            except ValueError:
                pass
            h = kernel_audio_lowpass_gaussian(size=size, sigma=sigma)
            filter_name = f"Low-Pass Gaussian (size={size}, σ={sigma})"

        elif pilihan == "3":
            h = kernel_audio_bassboost(size=101, sigma=20.0, gain=2.0)
            filter_name = "Bass Boost (Gaussian, gain=2.0)"

        elif pilihan == "4":
            size, sigma = 51, 10.0
            h = kernel_audio_highpass_from_lowpass(size=size, sigma=sigma)
            filter_name = f"High-Pass Filter (size={size}, σ={sigma})"

        elif pilihan == "5":
            h = kernel_audio_trebleboost(size=51, sigma=10.0, gain=2.0)
            filter_name = "Treble Boost (gain=2.0)"

        print(f"\n  Menerapkan {filter_name}...")
        print(f"  Panjang kernel: {len(h)} tap")

        # Konvolusi via FFT (dari convolution.py)
        y = convolve_fft(seg, h)
        y = normalize_audio(y)
        y_trimmed = trim_to_original_length(y, len(seg))

        # Hitung spektrum sebelum & sesudah untuk perbandingan
        print("  Menghitung spektrum FFT sebelum & sesudah...")
        seg_win = window_fn(seg)
        X_orig  = fft(seg_win)
        mag_orig = get_magnitude(X_orig)
        freqs_orig = get_frequency_axis(len(X_orig), sr)
        half = len(X_orig) // 2
        mag_orig  = mag_orig[:half]
        freqs_orig = freqs_orig[:half]

        y_win  = window_fn(y_trimmed)
        X_filt = fft(y_win)
        mag_filt  = get_magnitude(X_filt)
        freqs_filt = get_frequency_axis(len(X_filt), sr)
        mag_filt  = mag_filt[:half]
        freqs_filt = freqs_filt[:half]

        # Plot
        t_out = get_time_axis(y_trimmed, sr, start_sec)
        plot_filter_comparison(
            t_seg, seg, y_trimmed,
            freqs_orig, mag_orig,
            freqs_filt, mag_filt,
            filter_name=filter_name
        )
        pause()


# ═══════════════════════════════════════════════════════
#  MENU 10D — ECHO & REVERB
# ═══════════════════════════════════════════════════════

def menu_echo_reverb():
    while True:
        clear()
        header("10D. EFEK ECHO & REVERB")
        print("  Menggunakan kernel FIR + convolve_fft() dari convolution.py\n")
        print("  [1] Echo tunggal  (1 pantulan)")
        print("  [2] Reverb        (multi pantulan — efek ruangan)")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "0":
            break

        if pilihan not in ("1", "2"):
            print("  Input tidak valid.")
            input("  [Enter] lanjut...")
            continue

        result = _load_wav_interactive()
        if result is None:
            pause()
            continue

        segment, info, t, start_sec, end_sec = result

        # Downsample untuk kecepatan
        seg = list(segment)
        sr  = info.sample_rate
        if len(seg) > 16000:
            seg, sr = downsample(seg, sr, 16000)
            print(f"  (Downsampled ke {sr} Hz)")
        t_seg = get_time_axis(seg, sr, start_sec)

        if pilihan == "1":
            # Echo tunggal
            delay_sec = 0.3
            decay     = 0.5
            try:
                s = input(f"  Delay echo (detik) [default {delay_sec}]: ").strip()
                if s: delay_sec = float(s)
                s = input(f"  Decay (0.0–1.0) [default {decay}]: ").strip()
                if s: decay = float(s)
            except ValueError:
                print("  Input tidak valid, pakai default.")

            # Pastikan delay tidak lebih panjang dari segmen
            max_delay = (len(seg) / sr) * 0.5
            if delay_sec >= max_delay:
                delay_sec = max_delay * 0.5
                print(f"  ⚠ Delay terlalu panjang, dikurangi ke {delay_sec:.2f}s")

            h = kernel_audio_echo(sr, delay_sec=delay_sec, decay=decay)
            effect_name = f"Echo (delay={delay_sec:.2f}s, decay={decay:.2f})"

            print(f"\n  Menerapkan {effect_name}...")
            print(f"  Panjang kernel: {len(h):,} tap")

            y = convolve_fft(seg, h)
            y = normalize_audio(y)

            t_out = get_time_axis(y, sr, start_sec)
            plot_echo_effect(t_seg, seg, t_out, y,
                             delay_sec=delay_sec, decay=decay,
                             signal_name=f"Segmen {start_sec:.1f}s–{end_sec:.1f}s")

        elif pilihan == "2":
            # Reverb (preset)
            print("\n  Preset reverb:")
            print("  [1] Room (ruangan kecil)")
            print("  [2] Hall (aula besar)")
            print("  [3] Cathedral (katedral)")
            preset = input("  Pilih preset [1-3, default=1]: ").strip()

            if preset == "2":
                delays = [0.02, 0.05, 0.08, 0.15, 0.25, 0.40]
                decays  = [0.80, 0.65, 0.55, 0.40, 0.28, 0.18]
                effect_name = "Reverb — Hall"
            elif preset == "3":
                delays = [0.03, 0.07, 0.13, 0.22, 0.35, 0.55, 0.80]
                decays  = [0.85, 0.72, 0.60, 0.48, 0.35, 0.22, 0.12]
                effect_name = "Reverb — Cathedral"
            else:
                delays = [0.02, 0.04, 0.08, 0.12, 0.20]
                decays  = [0.70, 0.55, 0.40, 0.30, 0.18]
                effect_name = "Reverb — Room"

            # Clamp delay agar tidak melebihi panjang segmen
            max_delay = (len(seg) / sr) * 0.6
            delays = [min(d, max_delay) for d in delays]

            h = kernel_audio_reverb(sr, delays_sec=delays, decays=decays)
            print(f"\n  Menerapkan {effect_name}...")
            print(f"  Panjang kernel: {len(h):,} tap")

            y = convolve_fft(seg, h)
            y = normalize_audio(y)

            # Gunakan echo visualizer (same structure)
            t_out = get_time_axis(y, sr, start_sec)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(13, 6))
            fig.suptitle(f"{effect_name} | Segmen {start_sec:.1f}s–{end_sec:.1f}s",
                         fontsize=13, fontweight="bold")
            axes[0].plot(t_seg, seg, color="#2196F3", linewidth=0.7, label="Original")
            axes[0].set_title("① Sinyal Asli  x[n]", fontsize=10)
            axes[0].set_ylabel("Amplitudo"); axes[0].set_ylim(-1.1, 1.1)
            axes[0].legend(fontsize=8)
            axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

            t_out_plot = t_out[:len(y)]
            axes[1].plot(t_out_plot, y[:len(t_out_plot)],
                         color="#9C27B0", linewidth=0.7, alpha=0.85,
                         label=effect_name)
            axes[1].set_title(f"② Sinyal dengan {effect_name}  y[n]", fontsize=10)
            axes[1].set_xlabel("Waktu (s)"); axes[1].set_ylabel("Amplitudo")
            axes[1].legend(fontsize=8)
            axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            plt.tight_layout(); plt.show()

        pause()


# ═══════════════════════════════════════════════════════
#  MENU UTAMA AUDIO
# ═══════════════════════════════════════════════════════

def menu_audio():
    while True:
        clear()
        header("10. AUDIO SIGNAL PROCESSING — Sinyal Suara 1D")
        print("  Input  : File WAV (dibaca dari scratch, pure Python)")
        print("  FFT    : transforms.py (FFT manual Cooley-Tukey)")
        print("  Filter : convolution.py (konvolusi via FFT)\n")
        print("  [A] Load & Eksplorasi File WAV")
        print("  [B] Spektrum FFT Audio")
        print("  [C] Filter Audio (Low-Pass / High-Pass)")
        print("  [D] Efek Echo & Reverb")
        print("  [0] Kembali ke Menu Utama")
        print()
        pilihan = input("  Pilih: ").strip().upper()

        if   pilihan == "A": menu_load_eksplorasi()
        elif pilihan == "B": menu_spektrum_fft()
        elif pilihan == "C": menu_filter_audio()
        elif pilihan == "D": menu_echo_reverb()
        elif pilihan == "0": break
        else:
            print("  Input tidak valid.")
            input("  [Enter] lanjut...")
