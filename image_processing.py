"""
image_processing.py
===================
Implementasi manual:
  - Konvolusi 2D (Direct Spatial Convolution)
  - FFT 2D (via row-column decomposition)
  - IFFT 2D
  - Filter frekuensi domain (Low-Pass, High-Pass, Band-Pass)
  - Magnitude & Phase Spectrum 2D

Tidak menggunakan numpy.fft atau scipy.
"""

import math
from transforms import fft, ifft, Complex, _next_power_of_2, _zero_pad


# ═══════════════════════════════════════════════════════
#  BAGIAN 1: KONVOLUSI 2D
# ═══════════════════════════════════════════════════════

def convolve2d(image, kernel, padding="zero"):
    """
    Konvolusi 2D langsung (direct spatial convolution).

    Rumus:
        y[r, c] = Σ_i Σ_j h[i,j] * x[r-i, c-j]

    Untuk gambar ini diimplementasi sebagai cross-correlation
    dengan kernel diflip (sesuai definisi konvolusi):
        y[r, c] = Σ_i Σ_j h[i,j] * x[r + i - pad_r, c + j - pad_c]

    Parameter:
        image   : matrix 2D input (list of list)
        kernel  : matrix 2D kernel filter
        padding : 'zero' (pad dengan 0) atau 'replicate' (pad tepi)

    Return:
        output : matrix 2D hasil konvolusi (ukuran sama dengan input)
    """
    rows, cols = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    pad_r = kh // 2
    pad_c = kw // 2

    # Buat gambar dengan padding
    padded = _pad_image(image, pad_r, pad_c, mode=padding)
    p_rows = rows + 2 * pad_r
    p_cols = cols + 2 * pad_c

    output = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            total = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    # Flip kernel untuk konvolusi sejati
                    flipped_i = kh - 1 - ki
                    flipped_j = kw - 1 - kj
                    total += kernel[flipped_i][flipped_j] * padded[r + ki][c + kj]
            output[r][c] = total

    return output


def _pad_image(image, pad_r, pad_c, mode="zero"):
    """Tambahkan padding ke gambar."""
    rows, cols = len(image), len(image[0])
    new_rows = rows + 2 * pad_r
    new_cols = cols + 2 * pad_c
    padded = [[0.0] * new_cols for _ in range(new_rows)]

    for r in range(rows):
        for c in range(cols):
            padded[r + pad_r][c + pad_c] = image[r][c]

    if mode == "replicate":
        # Tepi atas & bawah
        for r in range(pad_r):
            for c in range(new_cols):
                padded[r][c] = padded[pad_r][min(max(c, pad_c), pad_c + cols - 1)]
                padded[new_rows - 1 - r][c] = padded[new_rows - 1 - pad_r][min(max(c, pad_c), pad_c + cols - 1)]
        # Tepi kiri & kanan
        for r in range(new_rows):
            for c in range(pad_c):
                padded[r][c] = padded[r][pad_c]
                padded[r][new_cols - 1 - c] = padded[r][new_cols - 1 - pad_c]

    return padded


# ─────────────────────────────────────────────
#  KERNEL 2D STANDAR
# ─────────────────────────────────────────────

def kernel2d_average(size=3):
    """
    Kernel rata-rata (box filter / mean filter).
    Efek: smoothing / blur.
    h[i,j] = 1/(size*size)
    """
    val = 1.0 / (size * size)
    return [[val] * size for _ in range(size)]


def kernel2d_gaussian(size=5, sigma=1.0):
    """
    Kernel Gaussian 2D.
    h[i,j] = e^(-(i²+j²)/(2σ²))
    Dinormalisasi agar jumlah = 1.
    Efek: blur lebih halus dari box filter.
    """
    center = size // 2
    kernel = []
    total = 0.0
    for i in range(size):
        row = []
        for j in range(size):
            val = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
            row.append(val)
            total += val
        kernel.append(row)
    # Normalisasi
    return [[v / total for v in row] for row in kernel]


def kernel2d_sobel_x():
    """
    Kernel Sobel horizontal — deteksi tepi vertikal.
    h = [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    """
    return [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ]


def kernel2d_sobel_y():
    """
    Kernel Sobel vertikal — deteksi tepi horizontal.
    h = [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    """
    return [
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ]


def kernel2d_sharpen():
    """
    Kernel sharpening.
    h = [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]]
    """
    return [
        [ 0.0, -1.0,  0.0],
        [-1.0,  5.0, -1.0],
        [ 0.0, -1.0,  0.0]
    ]


def kernel2d_laplacian():
    """
    Kernel Laplacian — deteksi tepi isotropik.
    h = [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    """
    return [
        [ 0.0,  1.0,  0.0],
        [ 1.0, -4.0,  1.0],
        [ 0.0,  1.0,  0.0]
    ]


def edge_magnitude(gx, gy):
    """
    Gabungkan Gx dan Gy menjadi magnitude tepi.
    |G| = sqrt(Gx² + Gy²)
    """
    rows, cols = len(gx), len(gx[0])
    result = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            result[r][c] = math.sqrt(gx[r][c] ** 2 + gy[r][c] ** 2)
    return result


# ═══════════════════════════════════════════════════════
#  BAGIAN 2: FFT 2D & IFFT 2D
# ═══════════════════════════════════════════════════════

def fft2d(image):
    """
    FFT 2D menggunakan dekomposisi baris-kolom (row-column decomposition).

    Algoritma:
        1. FFT setiap baris  → hasil intermediate
        2. FFT setiap kolom dari hasil intermediate
        → FFT 2D = FFT_kolom(FFT_baris(f))

    Separabilitas FFT 2D:
        F(u, v) = Σ_x Σ_y f(x,y) * e^(-j2πux/M) * e^(-j2πvy/N)
               = FFT_kolom { FFT_baris { f(x,y) } }

    Parameter:
        image : matrix 2D (list of list, nilai float)

    Return:
        F : matrix 2D dari bilangan Complex
    """
    rows, cols = len(image), len(image[0])

    # Step 1: FFT setiap baris
    row_fft = []
    for r in range(rows):
        row_fft.append(fft(image[r]))  # returns list of Complex

    # Pad kolom ke pangkat 2
    N_cols = _next_power_of_2(cols)

    # Step 2: FFT setiap kolom
    F = [[Complex(0, 0)] * N_cols for _ in range(rows)]
    for c in range(N_cols):
        # Ambil kolom ke-c dari row_fft (zero pad jika perlu)
        col = []
        for r in range(rows):
            if c < len(row_fft[r]):
                col.append(row_fft[r][c])
            else:
                col.append(Complex(0, 0))
        # FFT kolom — konversi ke float dulu via real/imag
        col_real = [v.real for v in col]
        col_imag = [v.imag for v in col]
        col_fft_r = _fft_complex_list(col)
        for r in range(rows):
            F[r][c] = col_fft_r[r]

    return F


def _fft_complex_list(x_complex):
    """
    FFT dari list bilangan Complex (bukan hanya real).
    Modifikasi dari FFT standar untuk input kompleks.
    """
    N = len(x_complex)
    N_pad = _next_power_of_2(N)
    if N_pad != N:
        x_complex = x_complex + [Complex(0, 0)] * (N_pad - N)
    return _fft_complex_recursive(x_complex)


def _fft_complex_recursive(x):
    """Rekursi FFT Cooley-Tukey untuk input kompleks."""
    N = len(x)
    if N == 1:
        return [x[0]]

    even = _fft_complex_recursive(x[0::2])
    odd  = _fft_complex_recursive(x[1::2])

    X = [Complex(0, 0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle)) * odd[k]
        X[k]           = even[k] + twiddle
        X[k + N // 2]  = even[k] - twiddle
    return X


def ifft2d(F):
    """
    IFFT 2D — inverse dari FFT 2D.

    Algoritma:
        1. IFFT setiap baris dari F
        2. IFFT setiap kolom dari hasil step 1

    Return:
        image : matrix 2D (nilai float, real part saja)
    """
    rows = len(F)
    cols = len(F[0])

    # Step 1: IFFT setiap baris
    row_ifft = []
    for r in range(rows):
        # IFFT baris dengan trik conjugate
        row_conj = [Complex(c.real, -c.imag) for c in F[r]]
        fft_row  = _fft_complex_recursive(row_conj)
        row_val  = [v.real / cols for v in fft_row]
        row_ifft.append([Complex(v, 0) for v in row_val])

    # Step 2: IFFT setiap kolom
    result = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        col = [row_ifft[r][c] for r in range(rows)]
        col_conj = [Complex(v.real, -v.imag) for v in col]
        fft_col  = _fft_complex_recursive(col_conj)
        for r in range(rows):
            result[r][c] = fft_col[r].real / rows

    return result


# ─────────────────────────────────────────────
#  MAGNITUDE & PHASE SPECTRUM 2D
# ─────────────────────────────────────────────

def magnitude_spectrum2d(F):
    """
    Ambil magnitude spektrum dari FFT 2D.
    |F(u,v)| = sqrt(Re² + Im²)

    Biasanya ditampilkan dalam skala log:
    log(1 + |F|) agar perbedaan terlihat jelas.
    """
    rows, cols = len(F), len(F[0])
    mag = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            mag[r][c] = math.log(1 + F[r][c].magnitude())
    return mag


def phase_spectrum2d(F):
    """Ambil spektrum fase dari FFT 2D."""
    rows, cols = len(F), len(F[0])
    phase = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            phase[r][c] = math.atan2(F[r][c].imag, F[r][c].real)
    return phase


def fftshift2d(matrix):
    """
    Geser komponen DC (frekuensi nol) ke tengah gambar.
    Sama seperti np.fft.fftshift — membuat spektrum lebih mudah divisualisasi.

    Cara kerja: geser setengah baris dan setengah kolom.
    """
    rows, cols = len(matrix), len(matrix[0])
    sr, sc = rows // 2, cols // 2
    shifted = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            new_r = (r + sr) % rows
            new_c = (c + sc) % cols
            shifted[new_r][new_c] = matrix[r][c]
    return shifted


def fftshift2d_complex(F):
    """fftshift untuk matrix Complex."""
    rows, cols = len(F), len(F[0])
    sr, sc = rows // 2, cols // 2
    shifted = [[Complex(0, 0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            new_r = (r + sr) % rows
            new_c = (c + sc) % cols
            shifted[new_r][new_c] = F[r][c]
    return shifted


# ─────────────────────────────────────────────
#  FILTER DOMAIN FREKUENSI
# ─────────────────────────────────────────────

def make_lowpass_mask(rows, cols, cutoff_ratio=0.1):
    """
    Buat mask low-pass filter lingkaran di domain frekuensi.
    Frekuensi dalam cutoff_ratio * min(rows, cols) diloloskan.

    Posisi DC sudah di tengah (setelah fftshift).
    """
    cutoff = cutoff_ratio * min(rows, cols) / 2
    cr, cc = rows // 2, cols // 2
    mask = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            dist = math.sqrt((r - cr) ** 2 + (c - cc) ** 2)
            mask[r][c] = 1.0 if dist <= cutoff else 0.0
    return mask


def make_highpass_mask(rows, cols, cutoff_ratio=0.1):
    """
    Buat mask high-pass filter (kebalikan dari low-pass).
    """
    lp = make_lowpass_mask(rows, cols, cutoff_ratio)
    return [[1.0 - lp[r][c] for c in range(cols)] for r in range(rows)]


def apply_frequency_filter(F_shifted, mask):
    """
    Terapkan mask filter ke spektrum FFT 2D yang sudah di-shift.

    F_filtered[u,v] = F_shifted[u,v] * mask[u,v]

    Parameter:
        F_shifted : matrix Complex (sudah fftshift)
        mask      : matrix float (0 atau 1)

    Return:
        F_filtered : matrix Complex hasil filtering
    """
    rows, cols = len(F_shifted), len(F_shifted[0])
    F_filtered = [[Complex(0, 0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            m = mask[r][c]
            F_filtered[r][c] = Complex(
                F_shifted[r][c].real * m,
                F_shifted[r][c].imag * m
            )
    return F_filtered


def ifftshift2d_complex(F):
    """Inverse fftshift untuk matrix Complex."""
    rows, cols = len(F), len(F[0])
    sr, sc = rows // 2, cols // 2
    shifted = [[Complex(0, 0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            new_r = (r - sr) % rows
            new_c = (c - sc) % cols
            shifted[new_r][new_c] = F[r][c]
    return shifted
