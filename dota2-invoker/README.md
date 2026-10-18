# Laporan Teknis Proyek 1 - Tipe 1: Dota 2 Invoker Hand Seals Detector (Pure OpenCV)

Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)  
Topik: *Mini Game Based on Hand Color & Geometric Contour Analysis*  
Berkas Utama: [`main.py`](main.py) | [`hand_detector.py`](hand_detector.py) | [`vfx_overlay.py`](vfx_overlay.py) | [`invoker_controller.py`](invoker_controller.py) | [`direct_input.py`](direct_input.py)

---

## 1. Pendahuluan dan Latar Belakang

*Dota 2* adalah game strategi berbasis MOBA (*Multiplayer Online Battle Arena*) di mana hero **Invoker** dikenal sebagai salah satu karakter paling kompleks. Invoker mengombinasikan tiga elemen dasar—**Quas (`Q`)**, **Wex (`W`)**, dan **Exort (`E`)**—menggunakan tombol **Invoke (`R`)** untuk meracik 10 mantra sihir berbeda, lalu melepaskannya melalui tombol **Quick Cast (`D` dan `F`)**.

Alih-alih menggunakan keyboard konvensional, proyek ini mengimplementasikan pengendali berbasis visi komputer murni (*pure computer vision without MediaPipe or deep learning models*) yang mengidentifikasi **Segel Tangan Ninja (*Naruto Jutsu Seals*)** secara *real-time*.

---

## 2. Metodologi Pengolahan Citra (Pure OpenCV)

Sistem dibangun berdasarkan materi Praktikum PCV Modul 1 sampai 5:

```text
[Webcam Stream BGR]
        │
        ▼
[Model Warna Multi-Kanal] ──► Konversi ke YCrCb & HSV
        │
        ▼
[Segmentasi Warna Kulit]  ──► Ambang Batas Adaptif (inRange)
        │
        ▼
[Operasi Morfologi Citra] ──► Opening (Erosi + Dilasi) lalu Closing (Dilasi + Erosi)
        │
        ▼
[Ekstraksi Kontur Utama]  ──► cv2.findContours (Hierarki Eksternal Terbesar)
        │
        ▼
[Analisis Geometri Bentuk]──► Aspect Ratio, Solidity, Convex Hull, Convexity Defects
        │
        ▼
[Temporal Debouncing]     ──► Hold Meter Thresholding (~0.2 detik konfirmasi)
        │
        ▼
[DirectX Key Injection]   ──► Windows API SendInput via Scancode Hardware
```

### 2.1. Model Warna dan Segmentasi Warna Kulit
Ruang warna RGB konvensional sangat rentan terhadap variasi intensitas pencahayaan (*luminance*). Oleh karena itu, sistem memisahkan krominans dan luminans menggunakan ruang warna **YCrCb** dan **HSV**:
* **YCrCb**: Memisahkan luminans ($Y$) dari komponen perbedaan warna merah ($Cr$) dan biru ($Cb$). Distribusi warna kulit manusia terkelompok secara ketat pada rentang:
  $$Cr \in [133, 173], \quad Cb \in [77, 127]$$
* **HSV**: Kanal *Hue* ($H$) mewakili corak warna kulit murni:
  $$H \in [0, 30], \quad S \in [25, 220], \quad V \in [40, 255]$$

Masker akhir merupakan irisan logika biner (*bitwise AND*):
$$M_{\text{raw}}(x, y) = M_{\text{YCrCb}}(x, y) \land M_{\text{HSV}}(x, y)$$

### 2.2. Operasi Morfologi Citra
Citra biner hasil thresholding dibersihkan menggunakan *Structuring Element* elips:
1. **Opening ($M \circ B$):** Erosi diikuti dilasi untuk mengeliminasi derau bintik putih di latar belakang:
   $$M \circ B = (M \ominus B) \oplus B$$
2. **Closing ($M \bullet B$):** Dilasi diikuti erosi untuk menutup rongga hitam di telapak tangan akibat bayangan sendi jari:
   $$M_{\text{clean}} = (M \circ B) \bullet B$$

### 2.3. Ekstraksi Fitur Geometri dan Pengenalan Segel
Kontur terbesar diekstraksi untuk menghitung deskriptor bentuk invariabel:

1. **Aspect Ratio ($AR$):**
   $$AR = \frac{h_{\text{bbox}}}{w_{\text{bbox}}}$$
2. **Solidity ($S$):** Rasio kepadatan area kontur terhadap area selubung cembung (*Convex Hull*):
   $$S = \frac{\text{Area}(C)}{\text{Area}(\text{ConvexHull}(C))}$$
3. **Convexity Defects:** Lekukan terdalam antara dua titik puncak selubung cembung:
   $$D = \frac{d_{\text{defect}}}{256.0}$$
4. **Puncak Runcing Vertikal (*Top Apex Ratio*):** Perbandingan lebar 15% kontur teratas terhadap lebar keseluruhan.

---

## 3. Pemetaan Segel Naruto ke Mekanik Invoker

| Nama Segel | Karakteristik Geometris OpenCV | Aksi Invoker | Tombol DirectX |
| :--- | :--- | :--- | :---: |
| **Segel Harimau (*Tiger / Tora*)** | Dua jari telunjuk tegak ke atas rapat ($AR \ge 1.35$, $S \in [0.65, 0.88]$, puncak runcing) | **Exort** (Elemen Api) | `E` |
| **Segel Ular (*Snake / Mi*)** | Dua telapak tangan menempel rapat tegak ($AR \ge 1.20$, $S \ge 0.84$, sedikit lekukan) | **Quas** (Elemen Es) | `Q` |
| **Segel Silang (*Cross / Kage Bunshin*)** | Dua jari menyilang tanda '+' ($0.75 \le AR \le 1.30$, $0.54 \le S \le 0.78$) | **Wex** (Elemen Petir) | `W` |
| **Segel Kambing / Kepal (*Ram / Fist*)** | Dua tangan mengepal menyatu rapat ($0.80 \le AR \le 1.25$, $S \ge 0.83$, 0 lekukan) | **Invoke** (Racik Mantra) | `R` |
| **Telapak Terbuka (*Open Palm*)** | 5 Jari mengembang menghadap kamera ($Defects \ge 3$, $S < 0.72$) | **Quick Cast Slot 1** | `D` |
| **Dua Telapak Terbuka (*Dual Palm*)** | Dua telapak tangan terbuka menyamping ($w > h \times 1.35$, area besar) | **Quick Cast Slot 2** | `F` |

---

## 4. Arsitektur DirectX Hardware Input (`direct_input.py`)

Game modern berbasis Source 2 Engine (Dota 2) memblokir fungsi injeksi keystroke tingkat tinggi OS (`pyautogui`) demi alasan anticheat dan arsitektur rendering DirectX 11/Vulkan.

Modul `direct_input.py` menggunakan Windows API `SendInput` tingkat rendah via `ctypes` dengan mengirimkan **Hardware Scancodes**:
* `KEYEVENTF_SCANCODE = 0x0008`
* Pemetaan scancode perangkat keras:
  - `Q = 0x10`, `W = 0x11`, `E = 0x12`, `R = 0x13`
  - `D = 0x20`, `F = 0x21`

---

## 5. Fitur Interaktif dan Petunjuk Eksekusi

### 5.1. Menjalankan Program
Pastikan webcam terhubung, lalu jalankan:
```bash
python project-1/dota2-invoker/main.py
```
Aplikasi akan membuka jendela mandiri berjudul **`Tugas Game - Dota 2 Invoker`**.

### 5.2. Kontrol Keyboard Interaktif
* **`C` (Calibrate)**: Masuk ke mode kalibrasi warna kulit adaptif. Posisikan tangan di dalam kotak kuning, lalu tekan `C` kembali untuk mengunci ambang batas pencahayaan ruang kelas.
* **`I` (DirectInput Toggle)**: Beralih antara mode **LIVE DIRECTX** (tombol dikirim langsung ke Dota 2) dan **SIMULASI** (hanya visual di layar).
* **`M` (Mask PiP)**: Menampilkan/menyembunyikan layar kecil (*Picture-in-Picture*) masker biner hasil morfologi di pojok kanan atas.
* **`R` (Reset Orbs)**: Mengembalikan komposisi orb ke `[E, E, E]`.
* **`Q` / `ESC`**: Keluar dari aplikasi.
