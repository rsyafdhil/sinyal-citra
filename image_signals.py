"""
image_signals.py
================
Generator sinyal 2D (gambar) dari scratch dan loader dari file.
Tidak menggunakan numpy atau scipy.

Representasi gambar: list of list (matrix 2D)
    image[row][col] = nilai grayscale (0.0 – 255.0)
"""

import math


# ─────────────────────────────────────────────
#  HELPER: Operasi Matrix 2D Manual
# ─────────────────────────────────────────────

def make_matrix(rows, cols, value=0.0):
    """Buat matrix 2D dengan nilai awal."""
    return [[value] * cols for _ in range(rows)]


def matrix_shape(image):
    """Return (rows, cols) dari matrix 2D."""
    return len(image), len(image[0])


def matrix_copy(image):
    """Salin matrix 2D."""
    return [row[:] for row in image]


def clamp(val, lo=0.0, hi=255.0):
    """Batasi nilai antara lo dan hi."""
    return max(lo, min(hi, val))


def normalize_matrix(image):
    """
    Normalisasi nilai pixel ke rentang 0–255.
    Berguna setelah filtering agar nilai tidak overflow.
    """
    rows, cols = matrix_shape(image)
    flat = [image[r][c] for r in range(rows) for c in range(cols)]
    min_val = min(flat)
    max_val = max(flat)
    if max_val == min_val:
        return make_matrix(rows, cols, 0.0)
    result = make_matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            result[r][c] = (image[r][c] - min_val) / (max_val - min_val) * 255.0
    return result


# ─────────────────────────────────────────────
#  1. GAMBAR SINTETIS DARI SCRATCH
# ─────────────────────────────────────────────

def generate_rectangle(rows=64, cols=64, rect_h=20, rect_w=30):
    """
    Gambar dengan kotak putih di tengah latar hitam.
    Berguna untuk demo konvolusi & FFT 2D.
    """
    image = make_matrix(rows, cols, 0.0)
    r_start = (rows - rect_h) // 2
    c_start = (cols - rect_w) // 2
    for r in range(r_start, r_start + rect_h):
        for c in range(c_start, c_start + rect_w):
            image[r][c] = 255.0
    return image


def generate_circle(rows=64, cols=64, radius=20):
    """
    Gambar dengan lingkaran putih di tengah latar hitam.
    """
    image = make_matrix(rows, cols, 0.0)
    cr, cc = rows // 2, cols // 2
    for r in range(rows):
        for c in range(cols):
            if (r - cr) ** 2 + (c - cc) ** 2 <= radius ** 2:
                image[r][c] = 255.0
    return image


def generate_checkerboard(rows=64, cols=64, block_size=8):
    """
    Pola papan catur.
    Sinyal frekuensi tinggi — bagus untuk demo FFT 2D.
    """
    image = make_matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if ((r // block_size) + (c // block_size)) % 2 == 0:
                image[r][c] = 255.0
    return image


def generate_gradient(rows=64, cols=64, direction="horizontal"):
    """
    Gambar gradien (ramp).
    direction: 'horizontal', 'vertical', atau 'diagonal'
    """
    image = make_matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if direction == "horizontal":
                image[r][c] = c / (cols - 1) * 255.0
            elif direction == "vertical":
                image[r][c] = r / (rows - 1) * 255.0
            else:  # diagonal
                image[r][c] = (r + c) / (rows + cols - 2) * 255.0
    return image


def generate_sinusoidal_2d(rows=64, cols=64, freq_x=4, freq_y=4):
    """
    Sinyal sinusoidal 2D.
    f(r, c) = 128 + 127 * sin(2π * fx * c/W) * sin(2π * fy * r/H)

    Berguna untuk visualisasi spektrum FFT 2D yang jelas.
    """
    image = make_matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            val = 128 + 127 * (
                math.sin(2 * math.pi * freq_x * c / cols) *
                math.sin(2 * math.pi * freq_y * r / rows)
            )
            image[r][c] = clamp(val)
    return image


def generate_impulse_2d(rows=64, cols=64):
    """
    Impuls 2D (delta 2D) — satu titik putih di tengah.
    Konvolusi dengan ini menghasilkan sinyal asli (identitas).
    """
    image = make_matrix(rows, cols, 0.0)
    image[rows // 2][cols // 2] = 255.0
    return image


def add_noise(image, sigma=20.0):
    """
    Tambahkan noise Gaussian ke gambar.
    Berguna untuk demo low-pass filter.

    Implementasi Box-Muller transform untuk noise Gaussian.
    """
    import random
    rows, cols = matrix_shape(image)
    result = make_matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            # Box-Muller
            u1 = random.random() + 1e-10
            u2 = random.random()
            noise = sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            result[r][c] = clamp(image[r][c] + noise)
    return result


# ─────────────────────────────────────────────
#  2. LOAD GAMBAR DARI FILE
# ─────────────────────────────────────────────

def load_image_ppm(filepath):
    """
    Load gambar dari file PPM (P3 / P6 format).
    PPM adalah format gambar tanpa library eksternal.

    Return:
        image : matrix 2D grayscale (0–255)
        (rows, cols)
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    lines = data.decode('latin-1').split('\n')
    idx = 0

    # Baca header
    magic = lines[idx].strip(); idx += 1
    while lines[idx].strip().startswith('#'):
        idx += 1

    dims = lines[idx].strip().split(); idx += 1
    cols, rows = int(dims[0]), int(dims[1])
    max_val = int(lines[idx].strip()); idx += 1

    # Ambil pixel values
    pixel_str = ' '.join(lines[idx:])
    values = list(map(int, pixel_str.split()))

    image = make_matrix(rows, cols)
    if magic == 'P3':  # ASCII RGB
        for r in range(rows):
            for c in range(cols):
                base = (r * cols + c) * 3
                R = values[base]
                G = values[base + 1]
                B = values[base + 2]
                # Konversi ke grayscale: Y = 0.299R + 0.587G + 0.114B
                image[r][c] = clamp(0.299 * R + 0.587 * G + 0.114 * B)
    return image, (rows, cols)


def load_image_matplotlib(filepath):
    """
    Load gambar menggunakan matplotlib (PNG/JPG).
    Matplotlib sudah dipakai di visualizer, jadi sudah tersedia.

    Return:
        image : matrix 2D grayscale (0–255)
        (rows, cols)
    """
    import matplotlib.image as mpimg
    img = mpimg.imread(filepath)

    # Handle berbagai format (RGB, RGBA, Grayscale)
    if len(img.shape) == 3:
        # RGB atau RGBA → Grayscale
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        rows, cols = img.shape[0], img.shape[1]
        image = make_matrix(rows, cols)
        for r in range(rows):
            for c in range(cols):
                # Normalisasi: matplotlib baca float 0–1 untuk PNG
                scale = 255.0 if img.max() <= 1.0 else 1.0
                gray = 0.299 * R[r, c] + 0.587 * G[r, c] + 0.114 * B[r, c]
                image[r][c] = clamp(float(gray) * scale)
    else:
        # Sudah grayscale
        rows, cols = img.shape
        image = make_matrix(rows, cols)
        scale = 255.0 if img.max() <= 1.0 else 1.0
        for r in range(rows):
            for c in range(cols):
                image[r][c] = clamp(float(img[r, c]) * scale)

    return image, (rows, cols)


def resize_image(image, new_rows, new_cols):
    """
    Resize gambar dengan nearest-neighbor interpolation.
    Berguna untuk memperkecil gambar besar agar FFT 2D lebih cepat.
    """
    rows, cols = matrix_shape(image)
    result = make_matrix(new_rows, new_cols)
    for r in range(new_rows):
        for c in range(new_cols):
            src_r = int(r * rows / new_rows)
            src_c = int(c * cols / new_cols)
            result[r][c] = image[src_r][src_c]
    return result
