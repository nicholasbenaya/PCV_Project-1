# Laporan Teknis Proyek 1 - Tipe 3: Whack-A-Mole Virtual Mini-Game (Pure OpenCV)

Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)  
Topik: *Virtual Interactive Whack-A-Mole Game Based on HSV Skin Segmentation & Centroid Tracking*  
Berkas Utama: [`main.py`](main.py) | [`hand_tracker.py`](hand_tracker.py) | [`game_engine.py`](game_engine.py) | [`entities.py`](entities.py) | [`ui_renderer.py`](ui_renderer.py) | [`config.py`](config.py)

---

## 1. Pendahuluan

**Whack-A-Mole Virtual** adalah game interaktif berbasis visi komputer murni (*pure OpenCV & NumPy*) tanpa pustaka *deep learning* (tanpa MediaPipe, TensorFlow, atau PyTorch). Pemain mengontrol permainan secara *hands-free* di depan webcam: menggerakkan telapak tangan untuk memukul karakter "mole" yang muncul secara acak pada grid 6 lubang (2 baris $\times$ 3 kolom).

Sistem melacak titik tengah (centroid) tangan pemain secara *real-time* menggunakan segmentasi ruang warna HSV, filter morfologi matematika, dan perhitungan *spatial image moments*.

---

## 2. Arsitektur Modular & Desain Sistem

Kode dirancang dengan prinsip *Separation of Concerns* (SoC) dan *Clean Architecture* agar mudah dipelihara (*highly maintainable*):

```text
whack-a-mole/
│
├── config.py           # Parameter konstan: batas warna HSV, ukuran frame, durasi gameplay, warna UI
├── hand_tracker.py     # Modul Computer Vision: segmentasi warna kulit HSV, morfologi, kontur & centroid EMA
├── entities.py         # Model data objek: Hole, Mole, HitEffect, dan ScoreManager
├── ui_renderer.py      # Modul visualisasi OpenCV: HUD semi-transparan, mole kartun, timer radial, panel modal
├── game_engine.py      # State Machine (START, PLAYING, GAMEOVER), collision detection, dan lifecycle loop
├── main.py             # Entry point: penanganan kamera, event keyboard, dan integrasi argumen CLI
└── README.md           # Laporan teknis dan dokumentasi lengkap
```

### Diagram Alur Data Sistem

```text
 ┌──────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
 │ Frame Webcam │ ───> │ cv2.flip (Mode Cermin)  │ ───> │ Gaussian Blur (9 x 9) │
 └──────────────┘      └─────────────────────────┘      └───────────┬───────────┘
                                                                    │
 ┌──────────────────────┐      ┌───────────────────────┐            ▼
 │ Kontur Terbesar      │ <─── │ Morfologi Open/Close  │ <─── ┌─────────────────┐
 │ (Area >= 2200 px)    │      │ (Kernel Elips 5 x 5)  │      │ Dual-Range HSV  │
 └──────────┬───────────┘      └───────────────────────┘      │ Skin Masking    │
            │                                                 └─────────────────┘
            ▼
 ┌──────────────────────┐      ┌───────────────────────┐      ┌─────────────────┐
 │ Spatial Moments      │ ───> │ EMA Smoothing         │ ───> │ Game Engine     │
 │ (cX, cY Centroid)    │      │ (Alpha = 0.65)        │      │ Collision Check │
 └──────────────────────┘      └───────────────────────┘      └────────┬────────┘
                                                                       │
                               ┌───────────────────────┐               ▼
                               │ Output Display Window │ <─── ┌─────────────────┐
                               │ (60 FPS Interactive)  │      │ UI Renderer     │
                               └───────────────────────┘      │ HUD, Mole, Vfx  │
                                                              └─────────────────┘
```

---

## 3. Metodologi Pengolahan Citra Digital (PCD)

### 3.1. Segmentasi Warna Kulit HSV Dual-Range
Ruang warna HSV (*Hue, Saturation, Value*) memisahkan kromatisitas (warna) dari luminans (intensitas cahaya), menjadikannya lebih tahan terhadap fluktuasi pencahayaan dibandingkan RGB biasa:
* **Rentang 1 (Oranye–Kuning & Merah Muda):**
  $$H \in [0, 22], \quad S \in [35, 190], \quad V \in [60, 255]$$
* **Rentang 2 (Pembungkus Rona Merah Pekat / Wrap-Around):**
  $$H \in [170, 180], \quad S \in [35, 190], \quad V \in [60, 255]$$
* **Penggabungan Masker:**
  $$M_{\text{raw}} = M_1 \lor M_2$$

### 3.2. Operasi Morfologi Matematika
Masker biner hasil thresholding dibersihkan menggunakan elemen penstruktur elips berukuran $5 \times 5$ ($B$):
$$M_{\text{clean}} = \left( (M_{\text{raw}} \circ B) \bullet B \right) \oplus B$$
1. **Opening ($M \circ B = (M \ominus B) \oplus B$):** Mengeliminasi derau bintik putih kecil (*salt noise*) di latar belakang.
2. **Closing ($M \bullet B = (M \oplus B) \ominus B$):** Menutup celah gelap (*pepper noise*) di permukaan telapak tangan.
3. **Dilasi ($M \oplus B$):** Mempertegas batas tepi kontur tangan sebelum ekstraksi.

### 3.3. Ekstraksi Titik Berat (*Spatial Moments*)
Setelah kontur eksternal terbesar diekstraksi dan disaring dengan ambang batas minimal $\text{Area} \ge 2.200\text{ px}$, koordinat centroid $(cX, cY)$ dihitung menggunakan momen spasial citra:
$$cX_{\text{raw}} = \frac{m_{10}}{m_{00}}, \quad cY_{\text{raw}} = \frac{m_{01}}{m_{00}}$$
di mana:
$$m_{ji} = \sum_{x} \sum_{y} x^j y^i I(x, y)$$

### 3.4. Stabilisasi Posisi (*Exponential Moving Average - EMA*)
Untuk mengeliminasi getaran mikro (*jitter*) pada kursor kordinat tanpa menimbulkan kelambatan (*lag*), diterapkan filter EMA:
$$\mathbf{C}_t = \alpha \mathbf{C}_{\text{raw}} + (1 - \alpha) \mathbf{C}_{t-1}$$
dengan faktor pembobotan $\alpha = 0.65$.

---

## 4. Mekanika Permainan & Logika Tabrakan

1. **Grid Arena 2x3:**
   - 6 lubang galian diposisikan secara dinamis berdasarkan lebar ($W$) dan tinggi ($H$) frame kamera.
   - Posisi kolom: $X \in \{0.22W, 0.50W, 0.78W\}$.
   - Posisi baris: $Y \in \{0.42H, 0.76H\}$.
   - Radius lubang: $R = 0.11 \times \min(W, H)$.

2. **Siklus Hidup Mole:**
   - Mole muncul pada salah satu lubang acak dengan durasi kemunculan antara $1.5$ hingga $2.0$ detik.
   - Terdapat **indikator busur radial waktu** di sekeliling lubang yang terus berkurang seiring habisnya waktu.
   - Jika waktu habis tanpa dipukul, mole berpindah lubang secara otomatis ke lubang lain.

3. **Deteksi Pukulan (Hit Detection) & Debounce:**
   - Pukulan dinyatakan valid jika jarak Euclidean antara titik kursor tangan pemain $(cX, cY)$ dan pusat mole $(x_{\text{mole}}, y_{\text{mole}})$ kurang dari atau sama dengan radius lubang:
     $$\text{distance} = \sqrt{(cX - x_{\text{mole}})^2 + (cY - y_{\text{mole}})^2} \le R$$
   - **Debounce Cooldown ($0.4\text{ detik}$):** Mencegah pukulan ganda dalam satu gerakan ayunan tangan.
   - Pukulan yang sukses memicu animasi lingkaran ekspansi hijau dan teks mengambang `+1 HIT!`.

4. **Mesin Status Permainan (State Machine):**
   - `START`: Layar sambutan, panduan kontrol, dan pratinjau kursor reticle tangan.
   - `PLAYING`: Sesi berdurasi 30 detik dengan skor berjalan dan hitung mundur timer.
   - `GAMEOVER`: Modal hasil akhir, rekor skor tertinggi (*High Score*), dan opsi main kembali.

---

## 5. Fitur Bantuan: Picture-in-Picture (PiP) Mask View

Pemain dapat menekan tombol **`M`** untuk menampilkan jendela kecil *Picture-in-Picture* di pojok kanan bawah layar. Fitur ini menampilkan masker biner hasil segmentasi warna kulit secara langsung, sehingga pemain dapat memverifikasi dan menyesuaikan pencahayaan ruangan kelas/kamar secara optimal sebelum bermain.

---

## 6. Petunjuk Instalasi & Menjalankan

### Persyaratan Sistem
Pastikan pustaka Python telah terpasang:
```bash
pip install -r requirements.txt
```
*(Membutuhkan `opencv-python >= 4.8.0` dan `numpy >= 1.24.0`)*

### Menjalankan Game
Dari direktori utama repositori (`project-1`):
```bash
python whack-a-mole/main.py
```
Atau langsung dari dalam sub-folder `whack-a-mole`:
```bash
cd whack-a-mole
python main.py
```

### Opsi Argumen Baris Perintah (CLI)
```bash
python whack-a-mole/main.py --camera 0 --width 640 --height 480 --duration 45.0
```

### Kontrol Keyboard
| Tombol | Fungsi |
| :--- | :--- |
| **`SPASI`** | Memulai permainan dari layar awal (*Start Screen*) |
| **`R`** | Mengulang permainan (*Restart*) dari layar Game Over |
| **`M`** | Menyalakan / mematikan jendela Picture-in-Picture (PiP) Mask Kulit |
| **`Q`** / **`ESC`** | Keluar dari permainan |
