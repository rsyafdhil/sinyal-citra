"""
signals.py
==========
Modul untuk membuat sinyal-sinyal dasar dari scratch.
Tidak menggunakan fungsi sinyal bawaan numpy/scipy.
"""

import math


# ─────────────────────────────────────────────
#  HELPER: buat array manual (tanpa np.zeros dll)
# ─────────────────────────────────────────────

def linspace(start, stop, n):
    """Buat n titik dari start sampai stop (seperti np.linspace)."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def arange(start, stop, step=1):
    """Buat list bilangan dari start sampai stop dengan step (seperti np.arange)."""
    result = []
    val = start
    while val < stop:
        result.append(val)
        val += step
    return result


# ─────────────────────────────────────────────
#  1. DELTA / IMPULSE SIGNAL
# ─────────────────────────────────────────────

def delta_signal(n_range=(-10, 10), shift=0):
    """
    Sinyal impuls (delta Kronecker).
    δ[n - shift] = 1 jika n == shift, 0 selainnya.

    Parameter:
        n_range : tuple (start, stop) range index n
        shift   : posisi impuls

    Return:
        n, x : list index dan list nilai sinyal
    """
    n = list(range(n_range[0], n_range[1] + 1))
    x = [1 if i == shift else 0 for i in n]
    return n, x


# ─────────────────────────────────────────────
#  2. STEP SIGNAL
# ─────────────────────────────────────────────

def step_signal(n_range=(-10, 10), shift=0):
    """
    Sinyal step (fungsi Heaviside diskrit).
    u[n - shift] = 1 jika n >= shift, 0 selainnya.

    Parameter:
        n_range : tuple (start, stop) range index n
        shift   : posisi mulai step

    Return:
        n, x : list index dan list nilai sinyal
    """
    n = list(range(n_range[0], n_range[1] + 1))
    x = [1 if i >= shift else 0 for i in n]
    return n, x


# ─────────────────────────────────────────────
#  3. SINYAL SINUS
# ─────────────────────────────────────────────

def sine_signal(frequency=1.0, amplitude=1.0, phase=0.0,
                t_range=(0, 2), num_points=500):
    """
    Sinyal sinus kontinu.
    x(t) = A * sin(2π * f * t + φ)

    Parameter:
        frequency  : frekuensi (Hz)
        amplitude  : amplitudo
        phase      : fase (radian)
        t_range    : tuple (start, stop) waktu
        num_points : jumlah titik sampling

    Return:
        t, x : list waktu dan list nilai sinyal
    """
    t = linspace(t_range[0], t_range[1], num_points)
    x = [amplitude * math.sin(2 * math.pi * frequency * ti + phase) for ti in t]
    return t, x


# ─────────────────────────────────────────────
#  4. SINYAL COSINUS
# ─────────────────────────────────────────────

def cosine_signal(frequency=1.0, amplitude=1.0, phase=0.0,
                  t_range=(0, 2), num_points=500):
    """
    Sinyal cosinus kontinu.
    x(t) = A * cos(2π * f * t + φ)

    Parameter:
        frequency  : frekuensi (Hz)
        amplitude  : amplitudo
        phase      : fase (radian)
        t_range    : tuple (start, stop) waktu
        num_points : jumlah titik sampling

    Return:
        t, x : list waktu dan list nilai sinyal
    """
    t = linspace(t_range[0], t_range[1], num_points)
    x = [amplitude * math.cos(2 * math.pi * frequency * ti + phase) for ti in t]
    return t, x


# ─────────────────────────────────────────────
#  5. SINYAL EKSPONENSIAL
# ─────────────────────────────────────────────

def exponential_signal(alpha=1.0, amplitude=1.0,
                       t_range=(0, 5), num_points=500):
    """
    Sinyal eksponensial.
    x(t) = A * e^(-alpha * t)

    alpha > 0 → meluruh (decaying)
    alpha < 0 → tumbuh  (growing)

    Parameter:
        alpha      : laju peluruhan/pertumbuhan
        amplitude  : amplitudo awal
        t_range    : tuple (start, stop) waktu
        num_points : jumlah titik sampling

    Return:
        t, x : list waktu dan list nilai sinyal
    """
    t = linspace(t_range[0], t_range[1], num_points)
    x = [amplitude * math.exp(-alpha * ti) for ti in t]
    return t, x


# ─────────────────────────────────────────────
#  6. SINYAL DISKRIT (sampling dari sinyal kontinu)
# ─────────────────────────────────────────────

def discrete_sine(frequency=1.0, amplitude=1.0, phase=0.0,
                  sampling_rate=20, t_range=(0, 2)):
    """
    Sinyal sinus diskrit — hasil sampling sinyal kontinu.
    x[n] = A * sin(2π * f * n/fs + φ)

    Parameter:
        frequency     : frekuensi sinyal (Hz)
        amplitude     : amplitudo
        phase         : fase (radian)
        sampling_rate : frekuensi sampling fs (sampel per detik)
        t_range       : tuple (start, stop) waktu

    Return:
        n, x : list index sampel dan list nilai sinyal
    """
    total_samples = int((t_range[1] - t_range[0]) * sampling_rate)
    n = list(range(total_samples))
    x = [amplitude * math.sin(2 * math.pi * frequency * i / sampling_rate + phase)
         for i in n]
    return n, x


# ─────────────────────────────────────────────
#  7. SINYAL GABUNGAN (multi-frekuensi)
# ─────────────────────────────────────────────

def composite_signal(frequencies, amplitudes, t_range=(0, 2), num_points=500):
    """
    Sinyal gabungan dari beberapa komponen sinus.
    x(t) = Σ A_i * sin(2π * f_i * t)

    Parameter:
        frequencies : list frekuensi
        amplitudes  : list amplitudo (panjang sama dengan frequencies)
        t_range     : tuple (start, stop) waktu
        num_points  : jumlah titik

    Return:
        t, x : list waktu dan list nilai sinyal gabungan
    """
    t = linspace(t_range[0], t_range[1], num_points)
    x = []
    for ti in t:
        val = sum(amplitudes[i] * math.sin(2 * math.pi * frequencies[i] * ti)
                  for i in range(len(frequencies)))
        x.append(val)
    return t, x
