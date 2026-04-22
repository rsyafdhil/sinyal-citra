"""
audio_signals.py
================
Modul untuk membaca file WAV dari scratch (pure Python),
tanpa menggunakan scipy.io.wavfile, librosa, atau library audio lainnya.

Fitur:
  - Baca file WAV (PCM 8-bit, 16-bit, 24-bit, 32-bit)
  - Stereo → mono (downmix)
  - Ambil segmen (detik ke-berapa sampai detik ke-berapa)
  - Normalisasi sampel ke float [-1.0, 1.0]
  - Info metadata WAV (sample rate, durasi, channels, bit depth)
"""

import math
import struct


# ═══════════════════════════════════════════════════════
#  WAV FILE READER — DARI SCRATCH
# ═══════════════════════════════════════════════════════

class WavInfo:
    """Menyimpan metadata file WAV."""
    def __init__(self):
        self.sample_rate   = 44100      # Hz (mis. 44100)
        self.num_channels  = 1      # 1 = mono, 2 = stereo
        self.bit_depth     = 16     # 8, 16, 24, atau 32
        self.num_samples   = 10      # total sampel per channel
        self.duration_sec  = 5.0   # durasi dalam detik
        self.audio_format  = 0      # 1 = PCM, 3 = IEEE float

    def __repr__(self):
        return (
            f"WAV Info:\n"
            f"  Sample Rate   : {self.sample_rate} Hz\n"
            f"  Channels      : {self.num_channels} "
            f"({'Mono' if self.num_channels == 1 else 'Stereo'})\n"
            f"  Bit Depth     : {self.bit_depth}-bit\n"
            f"  Total Samples : {self.num_samples:,}\n"
            f"  Durasi        : {self.duration_sec:.2f} detik "
            f"({self.duration_sec/60:.2f} menit)"
        )


def _read_chunk_header(f):
    """Baca 8 byte: 4 byte ID + 4 byte ukuran chunk (little-endian)."""
    data = f.read(8)
    if len(data) < 8:
        return None, 0
    chunk_id   = data[:4].decode('latin-1')
    chunk_size = struct.unpack('<I', data[4:8])[0]
    return chunk_id, chunk_size


def read_wav(filepath):
    """
    Baca file WAV dari scratch.

    Struktur WAV:
        RIFF header (12 byte)
        └── fmt  chunk  (16–40 byte): metadata audio
        └── data chunk  (N byte): sampel audio

    Parameter:
        filepath : path ke file .wav

    Return:
        samples  : list float [-1.0, 1.0] (mono, semua sampel)
        info     : WavInfo object
    """
    info = WavInfo()

    with open(filepath, 'rb') as f:
        # ── RIFF Header ──────────────────────────────
        riff_id   = f.read(4).decode('latin-1')
        riff_size = struct.unpack('<I', f.read(4))[0]
        wave_id   = f.read(4).decode('latin-1')

        if riff_id != 'RIFF' or wave_id != 'WAVE':
            raise ValueError(f"Bukan file WAV valid: {riff_id} / {wave_id}")

        # ── Cari chunk fmt dan data ──────────────────
        fmt_found  = False
        data_found = False
        raw_samples = None

        while True:
            chunk_id, chunk_size = _read_chunk_header(f)
            if chunk_id is None:
                break

            if chunk_id == 'fmt ':
                # ── fmt chunk ────────────────────────
                fmt_data = f.read(chunk_size)
                info.audio_format  = struct.unpack('<H', fmt_data[0:2])[0]
                info.num_channels  = struct.unpack('<H', fmt_data[2:4])[0]
                info.sample_rate   = struct.unpack('<I', fmt_data[4:8])[0]
                # byte_rate        = struct.unpack('<I', fmt_data[8:12])[0]
                # block_align      = struct.unpack('<H', fmt_data[12:14])[0]
                info.bit_depth     = struct.unpack('<H', fmt_data[14:16])[0]
                fmt_found = True

            elif chunk_id == 'data':
                # ── data chunk ───────────────────────
                raw_data = f.read(chunk_size)
                raw_samples = raw_data
                data_found = True
                break  # data chunk biasanya terakhir

            else:
                # Skip chunk lain (LIST, INFO, dsb.)
                f.seek(chunk_size, 1)

        if not fmt_found:
            raise ValueError("Chunk 'fmt ' tidak ditemukan dalam file WAV.")
        if not data_found:
            raise ValueError("Chunk 'data' tidak ditemukan dalam file WAV.")

    # ── Decode sampel audio ──────────────────────────
    samples = _decode_samples(raw_samples, info.bit_depth,
                               info.num_channels, info.audio_format)

    info.num_samples  = len(samples)
    info.duration_sec = info.num_samples / info.sample_rate

    return samples, info


def _decode_samples(raw_data, bit_depth, num_channels, audio_format):
    """
    Decode bytes mentah → list float mono [-1.0, 1.0].

    Format yang didukung:
      - PCM 8-bit  : unsigned int, center = 128
      - PCM 16-bit : signed int
      - PCM 24-bit : signed int (3 bytes)
      - PCM 32-bit : signed int
      - IEEE Float 32-bit

    Multi-channel → downmix ke mono (rata-rata semua channel).
    """
    bytes_per_sample = bit_depth // 8
    frame_size       = bytes_per_sample * num_channels
    num_frames       = len(raw_data) // frame_size

    samples_mono = []

    for i in range(num_frames):
        frame_start = i * frame_size
        channel_vals = []

        for ch in range(num_channels):
            offset = frame_start + ch * bytes_per_sample
            raw    = raw_data[offset : offset + bytes_per_sample]

            if audio_format == 3:
                # IEEE Float 32-bit
                val = struct.unpack('<f', raw)[0]

            elif bit_depth == 8:
                # PCM 8-bit unsigned (0–255, center = 128)
                val = (struct.unpack('<B', raw)[0] - 128) / 128.0

            elif bit_depth == 16:
                # PCM 16-bit signed
                val = struct.unpack('<h', raw)[0] / 32768.0

            elif bit_depth == 24:
                # PCM 24-bit signed (3 bytes, little-endian)
                b0, b1, b2 = raw[0], raw[1], raw[2]
                int_val = b0 | (b1 << 8) | (b2 << 16)
                if int_val >= 0x800000:  # sign extend
                    int_val -= 0x1000000
                val = int_val / 8388608.0  # 2^23

            elif bit_depth == 32:
                # PCM 32-bit signed
                val = struct.unpack('<i', raw)[0] / 2147483648.0

            else:
                raise ValueError(f"Bit depth tidak didukung: {bit_depth}")

            channel_vals.append(val)

        # Downmix ke mono
        mono_val = sum(channel_vals) / num_channels
        samples_mono.append(mono_val)

    return samples_mono


# ═══════════════════════════════════════════════════════
#  EKSTRAK SEGMEN WAKTU
# ═══════════════════════════════════════════════════════

def get_segment(samples, info, start_sec=0.0, end_sec=None):
    """
    Ambil segmen audio dari detik start_sec sampai end_sec.

    Parameter:
        samples   : list float (seluruh audio)
        info      : WavInfo
        start_sec : titik awal (detik), default = 0
        end_sec   : titik akhir (detik), default = akhir file

    Return:
        segment   : list float (potongan sampel)
        actual_start, actual_end : detik aktual yang diambil
    """
    if end_sec is None:
        end_sec = info.duration_sec

    # Validasi & clamp
    start_sec = max(0.0, min(start_sec, info.duration_sec))
    end_sec   = max(start_sec, min(end_sec, info.duration_sec))

    start_idx = int(start_sec * info.sample_rate)
    end_idx   = int(end_sec   * info.sample_rate)

    segment = samples[start_idx:end_idx]
    actual_start = start_idx / info.sample_rate
    actual_end   = end_idx   / info.sample_rate

    return segment, actual_start, actual_end


def get_time_axis(segment, sample_rate, start_sec=0.0):
    """
    Buat sumbu waktu untuk segmen audio.

    Return:
        t : list float (waktu dalam detik untuk setiap sampel)
    """
    n = len(segment)
    t = [start_sec + i / sample_rate for i in range(n)]
    return t


# ═══════════════════════════════════════════════════════
#  DOWNSAMPLING (opsional, untuk FFT lebih cepat)
# ═══════════════════════════════════════════════════════

def downsample(samples, original_rate, target_rate):
    """
    Downsample sinyal dengan simple decimation (setiap N sampel diambil 1).

    CATATAN: Ini adalah decimation sederhana tanpa anti-aliasing filter.
    Untuk sinyal musik, cukup untuk visualisasi.

    Parameter:
        samples       : list float
        original_rate : sample rate asli (Hz)
        target_rate   : sample rate target (Hz)

    Return:
        downsampled : list float
        target_rate : int (sample rate baru)
    """
    if target_rate >= original_rate:
        return samples, original_rate

    factor = original_rate // target_rate
    downsampled = samples[::factor]
    actual_rate = original_rate // factor
    return downsampled, actual_rate


# ═══════════════════════════════════════════════════════
#  WINDOWING (untuk analisis spektral lebih akurat)
# ═══════════════════════════════════════════════════════

def apply_hann_window(samples):
    """
    Terapkan Hann window ke sampel sebelum FFT.

    Hann window mengurangi spectral leakage (kebocoran frekuensi)
    yang terjadi karena sinyal dipotong tiba-tiba.

    w[n] = 0.5 * (1 - cos(2π*n / (N-1)))
    """
    N = len(samples)
    windowed = []
    for n in range(N):
        w = 0.5 * (1 - math.cos(2 * math.pi * n / (N - 1)))
        windowed.append(samples[n] * w)
    return windowed


def apply_hamming_window(samples):
    """
    Terapkan Hamming window ke sampel.

    w[n] = 0.54 - 0.46 * cos(2π*n / (N-1))
    """
    N = len(samples)
    windowed = []
    for n in range(N):
        w = 0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1))
        windowed.append(samples[n] * w)
    return windowed


def apply_rectangular_window(samples):
    """
    Rectangular window (tidak ada windowing) — kembalikan sampel apa adanya.
    """
    return list(samples)


# ═══════════════════════════════════════════════════════
#  STATISTIK SINYAL AUDIO
# ═══════════════════════════════════════════════════════

def signal_stats(samples):
    """
    Hitung statistik dasar sinyal audio.

    Return:
        dict dengan key: min, max, mean, rms, peak_db
    """
    n     = len(samples)
    total = sum(samples)
    mean  = total / n
    rms   = math.sqrt(sum(s ** 2 for s in samples) / n)
    peak  = max(abs(s) for s in samples)

    # Peak dalam dBFS (dB relative to full scale)
    peak_db = 20 * math.log10(peak + 1e-10)

    return {
        "min"     : min(samples),
        "max"     : max(samples),
        "mean"    : mean,
        "rms"     : rms,
        "peak"    : peak,
        "peak_db" : peak_db,
        "n_samples": n,
    }
