"""
audio_filters.py
================
Kernel dan efek audio untuk konvolusi 1D.
Semua filter diimplementasikan manual, terintegrasi dengan
convolve_direct() dan convolve_fft() dari convolution.py.

Filter tersedia:
  - Moving Average (Low-Pass sederhana)
  - Gaussian Low-Pass
  - High-Pass (diferensial)
  - Echo
  - Reverb (multi-echo)
  - Treble Boost
  - Bass Boost (alias low-pass kuat)
"""

import math


# ═══════════════════════════════════════════════════════
#  LOW-PASS FILTERS (smoothing / bass)
# ═══════════════════════════════════════════════════════

def kernel_audio_lowpass_moving_average(size=31):
    """
    Low-pass filter moving average untuk audio.
    Ukuran lebih besar → cutoff frekuensi lebih rendah.

    Efek: menghaluskan sinyal, membuang frekuensi tinggi (treble).

    Parameter:
        size : panjang kernel (ganjil disarankan)

    Return:
        h : list float, sum = 1.0
    """
    return [1.0 / size] * size


def kernel_audio_lowpass_gaussian(size=51, sigma=10.0):
    """
    Low-pass filter Gaussian untuk audio — lebih halus dari moving average.

    Parameter:
        size  : panjang kernel (ganjil disarankan)
        sigma : standar deviasi (lebih besar = cutoff lebih rendah)

    Return:
        h : list float, ternormalisasi (sum = 1.0)
    """
    center = size // 2
    h = [math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(size)]
    total = sum(h)
    return [v / total for v in h]


def kernel_audio_bassboost(size=101, sigma=20.0, gain=2.0):
    """
    Bass boost: low-pass Gaussian dengan gain > 1.
    Frekuensi rendah diperkuat, frekuensi tinggi dilemahkan.

    Parameter:
        gain : faktor penguatan bass (mis. 2.0 = +6 dB)
    """
    h = kernel_audio_lowpass_gaussian(size, sigma)
    return [v * gain for v in h]


# ═══════════════════════════════════════════════════════
#  HIGH-PASS FILTERS (treble / edge)
# ═══════════════════════════════════════════════════════

def kernel_audio_highpass_simple():
    """
    High-pass filter sederhana (diferensial diskrit).
    h = [1, -1]

    Efek: menguatkan frekuensi tinggi, melemahkan bass.
    """
    return [1.0, -1.0]


def kernel_audio_highpass_from_lowpass(size=51, sigma=10.0):
    """
    High-pass filter dengan teknik spectral inversion:
    h_hp[n] = delta[n] - h_lp[n]

    Menghasilkan high-pass yang lebih baik dari diferensial sederhana.

    Parameter:
        size, sigma : parameter untuk low-pass Gaussian dasar
    """
    h_lp = kernel_audio_lowpass_gaussian(size, sigma)
    center = size // 2
    # Buat kernel dengan impuls di tengah dikurangi low-pass
    h_hp = [-v for v in h_lp]
    h_hp[center] += 1.0  # tambah delta di tengah
    return h_hp


def kernel_audio_trebleboost(size=51, sigma=10.0, gain=2.0):
    """
    Treble boost: high-pass dengan gain.
    h = gain * h_hp
    """
    h_hp = kernel_audio_highpass_from_lowpass(size, sigma)
    return [v * gain for v in h_hp]


# ═══════════════════════════════════════════════════════
#  ECHO & REVERB
# ═══════════════════════════════════════════════════════

def kernel_audio_echo(sample_rate, delay_sec=0.3, decay=0.5):
    """
    Kernel echo sederhana (single echo).

    Cara kerja:
        y[n] = x[n] + decay * x[n - delay_samples]

    Representasi sebagai FIR kernel:
        h[0]           = 1.0        (sinyal asli)
        h[delay_samples] = decay    (echo)
        semua lainnya  = 0.0

    Parameter:
        sample_rate : Hz (mis. 44100)
        delay_sec   : jarak echo dalam detik (mis. 0.3 = 300ms)
        decay       : amplitude echo (0 = tidak ada, 1 = sama kuat)

    Return:
        h : list float (kernel panjang = delay_samples + 1)
    """
    delay_samples = int(delay_sec * sample_rate)
    h = [0.0] * (delay_samples + 1)
    h[0]              = 1.0    # sinyal langsung
    h[delay_samples]  = decay  # echo
    return h


def kernel_audio_reverb(sample_rate, delays_sec=None, decays=None):
    """
    Kernel reverb — multiple echo yang meniru ruang akustik.

    Parameter:
        sample_rate : Hz
        delays_sec  : list float (mis. [0.02, 0.05, 0.1, 0.2, 0.35])
        decays      : list float (mis. [0.7, 0.5, 0.4, 0.3, 0.2])

    Return:
        h : list float (FIR kernel)
    """
    if delays_sec is None:
        delays_sec = [0.02, 0.05, 0.1, 0.2, 0.35]
    if decays is None:
        decays     = [0.70, 0.55, 0.40, 0.28, 0.18]

    max_delay = max(int(d * sample_rate) for d in delays_sec)
    h = [0.0] * (max_delay + 1)
    h[0] = 1.0  # sinyal langsung

    for delay_sec, decay in zip(delays_sec, decays):
        idx = int(delay_sec * sample_rate)
        h[idx] += decay

    return h


# ═══════════════════════════════════════════════════════
#  NORMALISASI OUTPUT
# ═══════════════════════════════════════════════════════

def normalize_audio(samples, target_peak=0.9):
    """
    Normalisasi amplitudo audio ke target_peak.
    Berguna setelah konvolusi agar output tidak clipping.

    Parameter:
        samples     : list float
        target_peak : nilai puncak yang diinginkan (default 0.9)

    Return:
        normalized : list float
    """
    peak = max(abs(s) for s in samples)
    if peak < 1e-10:
        return list(samples)
    scale = target_peak / peak
    return [s * scale for s in samples]


def trim_to_original_length(y, original_length):
    """
    Potong output konvolusi ke panjang sinyal asli.
    Konvolusi menghasilkan N + M - 1 sampel, ini memotongnya ke N.
    """
    return y[:original_length]
