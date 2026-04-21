"""
menu_2d.py
==========
Modul menu interaktif untuk pengolahan sinyal 2D.
Di-import oleh main.py sebagai Modul 6 & 7.

Topik:
  6. Sinyal 2D & Gambar Dasar
  7. Konvolusi 2D (Spatial Domain)
  8. FFT 2D & Filtering Frekuensi
"""

import os
from image_signals import (
    generate_rectangle, generate_circle, generate_checkerboard,
    generate_gradient, generate_sinusoidal_2d, generate_impulse_2d,
    add_noise, load_image_matplotlib, resize_image
)
from image_processing import (
    convolve2d,
    kernel2d_average, kernel2d_gaussian,
    kernel2d_sobel_x, kernel2d_sobel_y,
    kernel2d_sharpen, kernel2d_laplacian,
    edge_magnitude,
    fft2d, ifft2d,
    magnitude_spectrum2d, phase_spectrum2d,
    fftshift2d, fftshift2d_complex, ifftshift2d_complex,
    make_lowpass_mask, make_highpass_mask, apply_frequency_filter
)
from image_visualizer import (
    plot_image, plot_all_synthetic_images,
    plot_convolution2d, plot_convolution2d_comparison,
    plot_fft2d_pipeline, plot_fft2d_filter_pipeline,
    plot_denoising
)
from image_signals import normalize_matrix


# ═══════════════════════════════════════════════════════
#  HELPER UI (konsisten dengan main.py)
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
#  HELPER: Load gambar dari file atau generate default
# ═══════════════════════════════════════════════════════

def _load_or_generate(prompt_user=True, default_size=64):
    """
    Tanya pengguna: load file atau gunakan gambar sintetis?
    Return: matrix 2D grayscale
    """
    print("\n  Sumber gambar:")
    print("  [1] Gambar sintetis (generated)")
    print("  [2] Load dari file gambar (PNG/JPG)")
    pilihan = input("  Pilih [1/2]: ").strip()

    if pilihan == "2":
        filepath = input("  Masukkan path file gambar: ").strip()
        if not os.path.exists(filepath):
            print(f"  ⚠ File tidak ditemukan: {filepath}")
            print("  Menggunakan gambar sintetis sebagai gantinya.")
            return generate_rectangle(default_size, default_size)
        try:
            image, (rows, cols) = load_image_matplotlib(filepath)
            # Resize jika terlalu besar (FFT 2D manual lambat untuk gambar besar)
            if rows > 128 or cols > 128:
                print(f"  Gambar {rows}x{cols} → di-resize ke 64x64 agar FFT lebih cepat.")
                image = resize_image(image, 64, 64)
            print(f"  ✓ Gambar berhasil dimuat.")
            return image
        except Exception as e:
            print(f"  ⚠ Gagal load gambar: {e}")
            print("  Menggunakan gambar sintetis sebagai gantinya.")
            return generate_rectangle(default_size, default_size)
    else:
        # Menu pilih gambar sintetis
        print("\n  Pilih gambar sintetis:")
        print("  [1] Kotak (Persegi Panjang)")
        print("  [2] Lingkaran")
        print("  [3] Papan Catur (Checkerboard)")
        print("  [4] Gradien")
        print("  [5] Sinusoidal 2D")
        print("  [6] Impuls 2D (titik tunggal)")
        pilihan2 = input("  Pilih [1-6, default=1]: ").strip()

        size = default_size
        if   pilihan2 == "2": return generate_circle(size, size)
        elif pilihan2 == "3": return generate_checkerboard(size, size)
        elif pilihan2 == "4": return generate_gradient(size, size)
        elif pilihan2 == "5": return generate_sinusoidal_2d(size, size, freq_x=4, freq_y=4)
        elif pilihan2 == "6": return generate_impulse_2d(size, size)
        else:                 return generate_rectangle(size, size)


# ═══════════════════════════════════════════════════════
#  MODUL 6 — SINYAL 2D & GAMBAR DASAR
# ═══════════════════════════════════════════════════════

def menu_sinyal_2d():
    while True:
        clear()
        header("6. SINYAL 2D — Gambar Dasar & Representasi")
        print("  [1] Tampilkan semua gambar sintetis dasar")
        print("  [2] Gambar + versi ber-noise (Gaussian noise)")
        print("  [3] Load gambar dari file")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            sub_header("Gambar Sintetis Dasar")
            images_data = [
                {"image": generate_rectangle(64, 64),          "title": "Kotak (Persegi Panjang)"},
                {"image": generate_circle(64, 64, radius=25),  "title": "Lingkaran"},
                {"image": generate_checkerboard(64, 64, 8),    "title": "Papan Catur (Checkerboard)"},
                {"image": generate_gradient(64, 64, "horizontal"), "title": "Gradien Horizontal"},
                {"image": generate_sinusoidal_2d(64, 64, 4, 4),   "title": "Sinusoidal 2D (fx=4, fy=4)"},
                {"image": generate_impulse_2d(64, 64),             "title": "Impuls 2D  δ[x,y]"},
            ]
            plot_all_synthetic_images(images_data)
            pause()

        elif pilihan == "2":
            sub_header("Gambar + Gaussian Noise")
            original = generate_rectangle(64, 64)
            noisy_light = add_noise(original, sigma=15.0)
            noisy_heavy = add_noise(original, sigma=50.0)
            images_data = [
                {"image": original,     "title": "Gambar Asli"},
                {"image": noisy_light,  "title": "Noise Ringan (σ=15)"},
                {"image": noisy_heavy,  "title": "Noise Berat (σ=50)"},
            ]
            plot_all_synthetic_images(images_data)
            pause()

        elif pilihan == "3":
            sub_header("Load dari File")
            filepath = input("  Masukkan path file gambar (PNG/JPG): ").strip()
            if not os.path.exists(filepath):
                print(f"  ⚠ File tidak ditemukan: {filepath}")
            else:
                try:
                    image, (rows, cols) = load_image_matplotlib(filepath)
                    print(f"  ✓ Ukuran: {rows}x{cols}")
                    plot_image(image, title=f"Gambar dari File: {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"  ⚠ Error: {e}")
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 7 — KONVOLUSI 2D
# ═══════════════════════════════════════════════════════

def menu_konvolusi_2d():
    while True:
        clear()
        header("7. KONVOLUSI 2D — Spatial Domain Filtering")
        print("  [1] Smoothing — Box (Average) Filter")
        print("  [2] Smoothing — Gaussian Filter")
        print("  [3] Sharpening — Kernel Sharpening")
        print("  [4] Edge Detection — Sobel (Gx, Gy, |G|)")
        print("  [5] Edge Detection — Laplacian")
        print("  [6] Denoising: Noisy → Gaussian Filter")
        print("  [7] Perbandingan semua filter sekaligus")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            sub_header("Box Filter (Average / Smoothing)")
            image = _load_or_generate()
            kernel = kernel2d_average(size=5)
            result = convolve2d(image, kernel)
            plot_convolution2d(image, kernel, result,
                               kernel_name="Box 5×5", filter_name="Box (Average) Filter")
            pause()

        elif pilihan == "2":
            sub_header("Gaussian Filter (Smoothing)")
            image = _load_or_generate()
            kernel = kernel2d_gaussian(size=7, sigma=1.5)
            result = convolve2d(image, kernel)
            plot_convolution2d(image, kernel, result,
                               kernel_name="Gaussian 7×7 (σ=1.5)",
                               filter_name="Gaussian Filter")
            pause()

        elif pilihan == "3":
            sub_header("Sharpening Filter")
            image = _load_or_generate()
            kernel = kernel2d_sharpen()
            result = convolve2d(image, kernel)
            plot_convolution2d(image, kernel, result,
                               kernel_name="Sharpen 3×3",
                               filter_name="Sharpening Filter")
            pause()

        elif pilihan == "4":
            sub_header("Sobel Edge Detection")
            image = _load_or_generate()
            kx = kernel2d_sobel_x()
            ky = kernel2d_sobel_y()
            gx = convolve2d(image, kx)
            gy = convolve2d(image, ky)
            mag = edge_magnitude(gx, gy)

            import matplotlib.pyplot as plt
            from image_signals import normalize_matrix
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle("Sobel Edge Detection", fontsize=13, fontweight="bold")
            axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("① Gambar Asli"); axes[0].axis("off")
            axes[1].imshow(normalize_matrix(gx), cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("② Gx (Tepi Vertikal)"); axes[1].axis("off")
            axes[2].imshow(normalize_matrix(gy), cmap="gray", vmin=0, vmax=255)
            axes[2].set_title("③ Gy (Tepi Horizontal)"); axes[2].axis("off")
            axes[3].imshow(normalize_matrix(mag), cmap="gray", vmin=0, vmax=255)
            axes[3].set_title("④ |G| = √(Gx²+Gy²)"); axes[3].axis("off")
            plt.tight_layout(); plt.show()
            pause()

        elif pilihan == "5":
            sub_header("Laplacian Edge Detection")
            image = _load_or_generate()
            kernel = kernel2d_laplacian()
            result = convolve2d(image, kernel)
            plot_convolution2d(image, kernel, result,
                               kernel_name="Laplacian 3×3",
                               filter_name="Laplacian Edge Detection")
            pause()

        elif pilihan == "6":
            sub_header("Denoising: Noisy → Gaussian Filter")
            print("  Membuat gambar kotak dengan noise berat...")
            original = generate_rectangle(64, 64)
            noisy    = add_noise(original, sigma=40.0)
            kernel   = kernel2d_gaussian(size=7, sigma=2.0)
            denoised = convolve2d(noisy, kernel)
            plot_denoising(original, noisy, denoised, method="Gaussian Filter 7×7")
            pause()

        elif pilihan == "7":
            sub_header("Perbandingan Semua Filter")
            image = _load_or_generate()
            results_data = [
                {"result": convolve2d(image, kernel2d_average(5)),          "title": "Box Filter 5×5"},
                {"result": convolve2d(image, kernel2d_gaussian(7, 1.5)),    "title": "Gaussian 7×7"},
                {"result": convolve2d(image, kernel2d_sharpen()),           "title": "Sharpen 3×3"},
                {"result": edge_magnitude(
                    convolve2d(image, kernel2d_sobel_x()),
                    convolve2d(image, kernel2d_sobel_y())
                ),                                                           "title": "Sobel |G|"},
                {"result": convolve2d(image, kernel2d_laplacian()),         "title": "Laplacian"},
            ]
            plot_convolution2d_comparison(image, results_data)
            pause()

        elif pilihan == "0":
            break


# ═══════════════════════════════════════════════════════
#  MODUL 8 — FFT 2D & FILTERING FREKUENSI
# ═══════════════════════════════════════════════════════

def menu_fft2d():
    while True:
        clear()
        header("8. FFT 2D — Transformasi & Filtering Frekuensi")
        print("  [1] Pipeline FFT 2D: Asli → Spektrum → Rekonstruksi")
        print("  [2] Spektrum berbagai gambar sintetis")
        print("  [3] Low-Pass Filter (domain frekuensi)")
        print("  [4] High-Pass Filter (domain frekuensi)")
        print("  [0] Kembali")
        print()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            sub_header("Pipeline FFT 2D Lengkap")
            image = _load_or_generate()
            print("  Menghitung FFT 2D... (sebentar)")
            F        = fft2d(image)
            F_shift  = fftshift2d_complex(F)
            mag      = magnitude_spectrum2d(F_shift)
            rec      = ifft2d(F)
            plot_fft2d_pipeline(image, mag, rec, signal_name="Gambar 2D")
            pause()

        elif pilihan == "2":
            sub_header("Spektrum FFT 2D Berbagai Gambar")
            images = [
                (generate_rectangle(64, 64),            "Kotak"),
                (generate_checkerboard(64, 64, 8),      "Papan Catur"),
                (generate_sinusoidal_2d(64, 64, 4, 4),  "Sinusoidal 2D"),
                (generate_impulse_2d(64, 64),            "Impuls 2D"),
            ]
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle("FFT 2D: Gambar Asli vs Spektrum Magnitude",
                         fontsize=14, fontweight="bold")
            for i, (img, title) in enumerate(images):
                print(f"  Menghitung FFT 2D untuk {title}...")
                F       = fft2d(img)
                F_shift = fftshift2d_complex(F)
                mag     = magnitude_spectrum2d(F_shift)
                mag_n   = normalize_matrix(mag)

                axes[0][i].imshow(img, cmap="gray", vmin=0, vmax=255)
                axes[0][i].set_title(title, fontsize=11)
                axes[0][i].axis("off")

                axes[1][i].imshow(mag_n, cmap="magma")
                axes[1][i].set_title(f"Spektrum |F(u,v)|", fontsize=11)
                axes[1][i].axis("off")

            plt.tight_layout()
            plt.show()
            pause()

        elif pilihan == "3":
            sub_header("Low-Pass Filter — Domain Frekuensi")
            image = _load_or_generate()
            print("  Menghitung FFT 2D...")
            rows, cols = len(image), len(image[0])
            F       = fft2d(image)
            F_shift = fftshift2d_complex(F)
            mag_orig = magnitude_spectrum2d(F_shift)

            # Terapkan Low-Pass Mask
            mask    = make_lowpass_mask(rows, len(F[0]), cutoff_ratio=0.15)
            F_filt  = apply_frequency_filter(F_shift, mask)
            mag_filt = magnitude_spectrum2d(F_filt)

            # Kembalikan ke spatial domain
            F_back  = ifftshift2d_complex(F_filt)
            result  = ifft2d(F_back)

            plot_fft2d_filter_pipeline(image, mag_orig, F_filt, mag_filt,
                                        result, filter_name="Low-Pass")
            pause()

        elif pilihan == "4":
            sub_header("High-Pass Filter — Domain Frekuensi")
            image = _load_or_generate()
            print("  Menghitung FFT 2D...")
            rows, cols = len(image), len(image[0])
            F       = fft2d(image)
            F_shift = fftshift2d_complex(F)
            mag_orig = magnitude_spectrum2d(F_shift)

            # Terapkan High-Pass Mask
            mask    = make_highpass_mask(rows, len(F[0]), cutoff_ratio=0.1)
            F_filt  = apply_frequency_filter(F_shift, mask)
            mag_filt = magnitude_spectrum2d(F_filt)

            F_back  = ifftshift2d_complex(F_filt)
            result  = ifft2d(F_back)

            plot_fft2d_filter_pipeline(image, mag_orig, F_filt, mag_filt,
                                        result, filter_name="High-Pass")
            pause()

        elif pilihan == "0":
            break
