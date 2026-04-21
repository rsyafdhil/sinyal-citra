"""
convolution.py
==============
Implementasi manual:
  - Konvolusi Langsung (Direct Convolution) — O(N*M)
  - Konvolusi via FFT                       — O(N log N)

Tidak menggunakan np.convolve atau scipy.signal.
"""

from transforms import fft, ifft, get_magnitude, Complex, _next_power_of_2, _zero_pad


# ─────────────────────────────────────────────
#  1. KONVOLUSI LANGSUNG — O(N*M)
# ─────────────────────────────────────────────

def convolve_direct(x, h):
    """
    Konvolusi linear langsung (direct/brute-force).

    Rumus:
        y[n] = Σ_{k=0}^{M-1} h[k] * x[n - k]

    Parameter:
        x : list sinyal input
        h : list kernel / impuls respons sistem

    Return:
        y : list hasil konvolusi, panjang = len(x) + len(h) - 1
    """
    N = len(x)
    M = len(h)
    L = N + M - 1  # panjang output

    y = [0.0] * L

    for n in range(L):
        for k in range(M):
            # Pastikan index tidak keluar batas
            x_idx = n - k
            if 0 <= x_idx < N:
                y[n] += h[k] * x[x_idx]

    return y


# ─────────────────────────────────────────────
#  2. KONVOLUSI VIA FFT — O(N log N)
# ─────────────────────────────────────────────

def convolve_fft(x, h):
    """
    Konvolusi linear menggunakan FFT (Convolution Theorem).

    Teorema Konvolusi:
        y = x * h  ↔  Y[k] = X[k] * H[k]

    Langkah:
        1. Pad x dan h ke panjang L = len(x) + len(h) - 1
           lalu round up ke pangkat 2
        2. FFT(x) dan FFT(h)
        3. Kalikan di domain frekuensi: Y = X * H
        4. IFFT(Y) → y

    Parameter:
        x : list sinyal input
        h : list kernel / impuls respons

    Return:
        y : list hasil konvolusi (panjang = len(x) + len(h) - 1)
    """
    N = len(x)
    M = len(h)
    L = N + M - 1                        # panjang output linear
    N_fft = _next_power_of_2(L)          # pad ke pangkat 2

    # Zero padding
    x_pad = _zero_pad(list(x), N_fft)
    h_pad = _zero_pad(list(h), N_fft)

    # FFT kedua sinyal
    X = fft(x_pad)
    H = fft(h_pad)

    # Perkalian di domain frekuensi
    Y = [X[k] * H[k] for k in range(N_fft)]

    # IFFT balik ke domain waktu
    y_full = ifft(Y)

    # Potong ke panjang L yang valid
    y = y_full[:L]

    return y


# ─────────────────────────────────────────────
#  3. HELPER: Buat Kernel Umum
# ─────────────────────────────────────────────

def kernel_moving_average(size=5):
    """
    Kernel rata-rata bergerak (low-pass filter sederhana).
    h[n] = 1/M untuk n = 0..M-1

    Efek: menghaluskan sinyal (smoothing).
    """
    return [1.0 / size] * size


def kernel_impulse():
    """
    Kernel impuls — konvolusi dengan ini tidak mengubah sinyal.
    h = [1]
    """
    return [1.0]


def kernel_derivative():
    """
    Kernel diferensial sederhana — seperti high-pass filter.
    h = [1, -1]
    Efek: mendeteksi perubahan (edge detection pada sinyal 1D).
    """
    return [1.0, -1.0]


def kernel_gaussian(size=11, sigma=2.0):
    """
    Kernel Gaussian — low-pass filter yang lebih halus.
    h[n] = e^(-(n - center)^2 / (2*sigma^2))
    Lalu dinormalisasi agar jumlah = 1.

    Parameter:
        size  : panjang kernel (ganjil disarankan)
        sigma : standar deviasi Gaussian
    """
    import math
    center = size // 2
    h = [math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(size)]
    total = sum(h)
    h = [v / total for v in h]
    return h
