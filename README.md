# Praktikum Pengolahan Citra dan Visi Komputer (PCV) - Project 1: Mini Game Based on Hand Color

Repositori ini memuat implementasi **Project 1: Mini Game Based on Hand Color** mata kuliah Praktikum Pengolahan Citra dan Visi Komputer (PCV). Seluruh pemrosesan citra, segmentasi warna, morfologi, analisis geometri kontur, dan rendering antarmuka game dibangun secara murni (*from scratch*) menggunakan **OpenCV (`cv2`) dan NumPy**, tanpa menggunakan pustaka *deep learning* atau pelacak sendi instan (*No MediaPipe, No YOLO*).

---

## 📁 Struktur Direktori Repositori

Proyek ini terbagi menjadi dua sub-folder mandiri agar arsitektur dan laporannya tidak saling tercampur:

```text
project-1/
│
├── dota2-invoker/                      # [TIPE 1: Game yang Sudah Ada - Dota 2]
│   ├── direct_input.py                 # Injeksi Hardware Scancode DirectInput (Windows SendInput)
│   ├── hand_detector.py                # Segmentasi Warna Kulit YCrCb/HSV & Klasifikasi Segel Tangan
│   ├── vfx_overlay.py                  # Efek Visual Lingkaran Sihir & Partikel Elemen
│   ├── invoker_controller.py           # Logika Invoker (3 Orbs, 10 Resep Mantra, Slot D & F)
│   ├── main.py                         # Program Utama (Jendela: "Tugas Game - Dota 2 Invoker")
│   └── README.md                       # Laporan Teknis Lengkap Tipe 1
│
├── custom-minigame/                    # [TIPE 2: Custom Game Buatan Sendiri - Shinobi Battle]
│   ├── hand_detector.py                # Modul Deteksi Segel Tangan Responsif Real-Time
│   ├── entities.py                     # Entitas Pemain, Demon Boss, Proyektil Sihir, & Partikel
│   ├── game_engine.py                  # Game Engine 2D 60 FPS & Renderer Arena Murni OpenCV
│   ├── main.py                         # Program Utama (Jendela: "Tugas Game - Shinobi Jutsu Battle")
│   └── README.md                       # Laporan Teknis Lengkap Tipe 2
│
├── whack-a-mole/                       # [TIPE 3: Whack-A-Mole Virtual - Centroid Tracking]
│   ├── config.py                       # Parameter Segmentasi HSV, Kamera, dan Gameplay
│   ├── hand_tracker.py                 # Segmentasi Kulit HSV & Spatial Moments Centroid (EMA)
│   ├── entities.py                     # Entitas Mole, Hole 2x3, HitEffect, & ScoreManager
│   ├── ui_renderer.py                  # HUD Semi-Transparan, Mole Kartun, & PiP Mask View
│   ├── game_engine.py                  # State Machine (START, PLAYING, GAMEOVER) & Hit Detection
│   ├── main.py                         # Program Utama (Jendela: "Tugas Game - Whack-A-Mole Virtual")
│   └── README.md                       # Laporan Teknis Lengkap Tipe 3
│
├── requirements.txt                    # opencv-python & numpy
├── .gitignore
└── README.md
```

---

## 🎮 Ringkasan Tipe Game

### 1. Tipe 1: Dota 2 Invoker Hand Pose Controller (`dota2-invoker/`)
* **Konsep**: Mengontrol perapalan mantra hero Invoker di game Dota 2 secara *hands-free* menggunakan **Penghitungan Jari & Gestur Tangan (Convexity Defects & Geometry)**.
* **Mekanik Gestur**:
  * **1 Jari** $\rightarrow$ Menambah Orb **Quas (`Q`)** (Es)
  * **2 Jari** $\rightarrow$ Menambah Orb **Wex (`W`)** (Petir)
  * **3 Jari** $\rightarrow$ Menambah Orb **Exort (`E`)** (Api)
  * **4 Jari** $\rightarrow$ **Invoke (`R`)** (Meracik 10 mantra)
  * **5 Jari (Telapak Terbuka)** $\rightarrow$ **Quick Cast Spell 1 (`D`)**
  * **Kepal (Fist / 0 Jari)** $\rightarrow$ **Quick Cast Spell 2 (`F`)**
* **Eksekusi**:
  ```bash
  python dota2-invoker/main.py
  ```
* 📄 **Laporan Lengkap**: [Baca README Dota 2 Invoker](dota2-invoker/README.md)

---

### 2. Tipe 2: Shinobi Jutsu Battle Mini-Game (`custom-minigame/`)
* **Konsep**: Game pertarungan sihir 2D murni OpenCV yang menampilkan feed kamera pemain di sisi kiri dan arena pertempuran melawan *Demon Boss Susanoo* di sisi kanan dalam satu jendela terpadu.
* **Mekanik Jurus**:
  * **Segel Harimau** $\rightarrow$ **Katon: Goukakyuu!** (Bola api besar, 45 Damage)
  * **Segel Ular** $\rightarrow$ **Hyoton: Ice Barrier!** (Perisai es penangkal serangan + Pemulihan HP)
  * **Segel Silang** $\rightarrow$ **Raiton: Chidori Ray!** (Sinar kilat cepat, 35 Damage)
  * **Segel Kambing / Kepal** $\rightarrow$ **Fuuton: Rasenshuriken!** (Shuriken angin berputar, 70 Damage)
  * **Telapak Terbuka** $\rightarrow$ **Chakra Gatling!** (Tembakan beruntun 3 bola energi)
* **Eksekusi**:
  ```bash
  python custom-minigame/main.py
  ```
* 📄 **Laporan Lengkap**: [Baca README Custom Mini-Game](custom-minigame/README.md)

---

### 3. Tipe 3: Whack-A-Mole Virtual Mini-Game (`whack-a-mole/`)
* **Konsep**: Game interaktif arcade memukul karakter "mole" yang muncul secara acak di salah satu dari 6 lubang target (grid 2x3) di atas feed webcam cermin secara *real-time*.
* **Mekanik Permainan**:
  * **Pelacakan Centroid Tangan**: Segmentasi warna kulit HSV dual-range dan momen spasial $cX = m_{10}/m_{00}, cY = m_{01}/m_{00}$ dengan stabilisasi EMA smoothing.
  * **Deteksi Tabrakan (Hit Detection)**: Jarak Euclidean ke pusat mole aktif dengan jeda *debounce cooldown* 0,4 detik.
  * **Visual & Efek**: Timer radial mole, HUD skor & countdown 30s, animasi ledakan `+1 HIT!`, serta fitur PiP Mask View (`M`).
* **Eksekusi**:
  ```bash
  python whack-a-mole/main.py
  ```
* 📄 **Laporan Lengkap**: [Baca README Whack-A-Mole](whack-a-mole/README.md)

---

## 🔬 Fondasi Teori Pengolahan Citra Digital (Tanpa MediaPipe)

Kedua proyek mengimplementasikan teori dari praktikum PCV:
1. **Model Warna**: Segmentasi krominans $Cr$ & $Cb$ pada ruang warna **YCrCb** serta $H$ pada ruang warna **HSV** untuk kestabilan isolasi warna kulit di bawah berbagai kondisi lampu ruangan.
2. **Operasi Morfologi**: Penerapan *Opening* ($M \circ B$) untuk mengeliminasi derau latar belakang dan *Closing* ($M \bullet B$) untuk menutup lubang kontur di dalam telapak tangan.
3. **Analisis Geometri Kontur**: Ekstraksi *Aspect Ratio* ($h/w$), *Solidity* ($\text{Area} / \text{Hull Area}$), *Convexity Defects* (kedalaman celah antar jari), dan rasio puncak (*Top Apex Ratio*).
4. **Kalibrasi Adaptif (Tombol 'C')**: Fitur pengambilan sampel nilai $\mu$ dan $\sigma$ warna kulit secara instan untuk demo di ruang kelas (AJ401-402).

---

## 🚀 Persyaratan Sistem & Instalasi

```bash
pip install -r requirements.txt
```
Dependensi minimal:
* `opencv-python >= 4.8.0`
* `numpy >= 1.24.0`
* Sistem Operasi: Windows 10/11 (Mendukung Windows API `SendInput` untuk input DirectX).
