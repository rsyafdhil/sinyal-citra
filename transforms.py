"""
transforms.py
=============
Implementasi manual:
  - DFT  (Discrete Fourier Transform)
  - IDFT (Inverse DFT)
  - FFT  (Fast Fourier Transform — Cooley-Tukey Radix-2)
  - IFFT (Inverse FFT)

Tidak menggunakan np.fft atau scipy.fft.
"""

import math


# ─────────────────────────────────────────────
#  HELPER: Bilangan Kompleks Manual
# ─────────────────────────────────────────────

class Complex:
    """Representasi bilangan kompleks sederhana."""

    def __init__(self, real=0.0, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        # (a+bj)(c+dj) = (ac-bd) + (ad+bc)j
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def __repr__(self):
        return f"({self.real:.4f} + {self.imag:.4f}j)"


def exp_j(theta):
    """e^(jθ) = cos(θ) + j*sin(θ)  — Euler's formula."""
    return Complex(math.cos(theta), math.sin(theta))


# ─────────────────────────────────────────────
#  1. DFT — O(N²)
# ─────────────────────────────────────────────

def dft(x):
    """
    Discrete Fourier Transform (DFT) manual.

    Rumus:
        X[k] = Σ_{n=0}^{N-1} x[n] * e^(-j2πkn/N)

    Parameter:
        x : list sinyal input (real)

    Return:
        X : list Complex (spektrum frekuensi)
    """
    N = len(x)
    X = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(x[n], 0) * exp_j(angle)
        X.append(total)
    return X


# ─────────────────────────────────────────────
#  2. IDFT — Inverse DFT — O(N²)
# ─────────────────────────────────────────────

def idft(X):
    """
    Inverse Discrete Fourier Transform (IDFT) manual.

    Rumus:
        x[n] = (1/N) * Σ_{k=0}^{N-1} X[k] * e^(j2πkn/N)

    Parameter:
        X : list Complex (spektrum frekuensi)

    Return:
        x : list nilai real hasil IDFT
    """
    N = len(X)
    x = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + X[k] * exp_j(angle)
        x.append(total.real / N)
    return x


# ─────────────────────────────────────────────
#  3. FFT — Cooley-Tukey Radix-2 — O(N log N)
# ─────────────────────────────────────────────

def _next_power_of_2(n):
    """Cari pangkat 2 terdekat yang >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _zero_pad(x, target_len):
    """Pad sinyal dengan nol sampai panjang target_len."""
    return x + [0.0] * (target_len - len(x))


def fft(x):
    """
    Fast Fourier Transform (FFT) — Cooley-Tukey Radix-2 DIT.

    Algoritma divide & conquer:
      FFT(x) = FFT(x_even) + W * FFT(x_odd)
      dimana W = e^(-j2πk/N) (twiddle factor)

    Input otomatis di-pad ke pangkat 2 terdekat.

    Parameter:
        x : list sinyal input (real/float)

    Return:
        X : list Complex (spektrum frekuensi)
    """
    N_orig = len(x)
    N = _next_power_of_2(N_orig)

    # Zero padding kalau perlu
    if N != N_orig:
        x = _zero_pad(x, N)

    # Konversi ke Complex
    x_c = [Complex(val, 0) for val in x]
    return _fft_recursive(x_c)


def _fft_recursive(x):
    """Rekursi FFT Cooley-Tukey."""
    N = len(x)

    # Base case
    if N == 1:
        return [x[0]]

    # Pisah genap & ganjil
    even = _fft_recursive(x[0::2])
    odd  = _fft_recursive(x[1::2])

    # Gabungkan dengan twiddle factor
    X = [Complex(0, 0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = exp_j(angle) * odd[k]
        X[k]           = even[k] + twiddle
        X[k + N // 2]  = even[k] - twiddle

    return X


# ─────────────────────────────────────────────
#  4. IFFT — Inverse FFT
# ─────────────────────────────────────────────

def ifft(X):
    """
    Inverse Fast Fourier Transform (IFFT).

    Trik: IFFT(X) = conj(FFT(conj(X))) / N

    Parameter:
        X : list Complex (spektrum frekuensi)

    Return:
        x : list float (sinyal domain waktu)
    """
    N = len(X)

    # Konjugat input
    X_conj = [Complex(c.real, -c.imag) for c in X]

    # FFT dari konjugat
    x_conj = _fft_recursive(X_conj)

    # Konjugat hasil dan bagi N
    x = [c.real / N for c in x_conj]
    return x


# ─────────────────────────────────────────────
#  5. HELPER: Ambil magnitude & frekuensi
# ─────────────────────────────────────────────

def get_magnitude(X):
    """Ambil magnitude spektrum dari hasil DFT/FFT."""
    return [c.magnitude() for c in X]


def get_phase(X):
    """Ambil fase spektrum (dalam radian)."""
    return [c.phase() for c in X]


def get_frequency_axis(N, sampling_rate=1.0):
    """
    Buat sumbu frekuensi untuk plot spektrum.

    Parameter:
        N            : panjang sinyal
        sampling_rate: frekuensi sampling (Hz)

    Return:
        freqs : list frekuensi (Hz), hanya sisi positif (0 sampai fs/2)
    """
    freqs = [k * sampling_rate / N for k in range(N // 2)]
    return freqs
