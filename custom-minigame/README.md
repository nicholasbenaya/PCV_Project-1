# Laporan Teknis Proyek 1 - Tipe 2: Shinobi Jutsu Battle Mini-Game (Pure OpenCV)

Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)  
Topik: *Custom Mini Game Based on Hand Color, Morphology, & Shape Analysis*  
Berkas Utama: [`main.py`](main.py) | [`hand_detector.py`](hand_detector.py) | [`game_engine.py`](game_engine.py) | [`entities.py`](entities.py)

---

## 1. Pendahuluan

Proyek ini merupakan realisasi mandiri dari tugas **Mini Game Based on Hand Color** yang dibangun 100% menggunakan pustaka **OpenCV (`cv2`) dan NumPy** tanpa dependensi game engine eksternal (seperti Pygame/Unity) maupun model *deep learning* (seperti MediaPipe/YOLO).

Pemain berperan sebagai seorang *Shinobi* / penyihir petarung yang merapalkan **Segel Tangan Ninja (*Naruto Jutsu Seals*)** di depan kamera untuk menembakkan proyektil sihir berelemen ke arah monster *Susanoo Shadow Demon*.

---

## 2. Arsitektur Sistem Dual-Pane

Program menggabungkan dua lingkungan komputasi dalam satu kanvas terpadu beresolusi $1280 \times 720$:

```text
┌─────────────────────────────────┬────────────────────────────────────────────────────────┐
│   PANEL KAMERA PCV (480 x 720)   │              PANEL ARENA GAME (800 x 720)              │
│                                 │                                                        │
│ • Feed Kamera Real-time         │ • HUD Darah Shinobi (100 HP) & Boss (500 HP)           │
│ • Masker Kulit Morfologi (PiP)  │ • Karakter Shinobi Player (Avatar + Pelindung Es)      │
│ • Kontur & Selubung Cembung     │ • Boss Monster Susanoo (Animasi Melayang & Tembakan)   │
│ • Lingkaran Sihir Berputar      │ • Proyektil Sihir Real-time & Partikel Ledakan         │
│ • Hold Meter Pemicu Segel       │ • Teks Damage Floating & Screen Shake saat Benturan    │
└─────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. Metodologi Pengolahan Citra Digital

### 3.1. Segmentasi Warna Kulit Adaptif
Pengambilan segmentasi warna menggunakan model warna ganda:
* **YCrCb:** Memisahkan krominans $Cr$ dan $Cb$ untuk mengatasi perubahan intensitas cahaya ruangan kelas:
  $$Cr \in [133, 173], \quad Cb \in [77, 127]$$
* **HSV:** Membatasi *Hue* ($H \in [0, 30]$) dan saturasi untuk membedakan kulit dari objek latar belakang.
* **Fitur Kalibrasi Ruangan (Tombol 'C'):** Mengukur rata-rata ($\mu$) dan simpangan baku ($\sigma$) sampel warna kulit di area ROI tengah secara otomatis.

### 3.2. Operasi Morfologi Citra (Erosi & Dilasi)
Menggunakan elemen penstruktur elips berukuran $3 \times 3$ dan $7 \times 7$:
$$\text{Mask}_{\text{bersih}} = (M_{\text{raw}} \circ B_{\text{open}}) \bullet B_{\text{close}}$$
* *Opening*: Menghilangkan derau garam (*salt noise*) di latar belakang.
* *Closing*: Menutup lubang (*pepper noise*) di dalam telapak tangan.

### 3.3. Klasifikasi Geometri Bentuk Segel
1. **Rasio Aspek ($AR = h / w$):** Membedakan pose vertikal memanjang (*Tiger / Snake*) dari pose melebar.
2. **Kepadatan (*Solidity* $S = \text{Area} / \text{Hull Area}$):** Membedakan tangan tertutup rapat dari tangan dengan jari terbuka.
3. **Cacat Kecembungan (*Convexity Defects*):** Menghitung lekukan terdalam antar jari untuk mendeteksi telapak tangan terbuka (*Chakra Gatling*).

---

## 4. Mekanik Jurus dan Segel Tangan

| Nama Segel | Ciri Geometri Citra | Jurus yang Dikeluarkan | Efek Pertempuran |
| :--- | :--- | :--- | :--- |
| **TIGER (Harimau)** | $AR \ge 1.35$, Puncak Runcing, $S \in [0.65, 0.88]$ | **Katon: Goukakyuu!** | Bola api besar berkecepatan sedang, memberikan **45 Damage**. |
| **SNAKE (Ular / Doa)** | $AR \ge 1.20$, $S \ge 0.84$, $\le 2$ Cacat | **Hyoton: Ice Barrier!** | Mengaktifkan perisai es selama 5 detik (menangkis serangan boss) dan memulihkan **+20 HP**. |
| **CROSS (Silang)** | $0.75 \le AR \le 1.30$, $S \in [0.54, 0.78]$ | **Raiton: Chidori Ray!** | Kilat berkecepatan tinggi ($650\text{ px/s}$), memberikan **35 Damage**. |
| **RAM (Kambing / Kepal)** | $0.80 \le AR \le 1.25$, $S \ge 0.83$, 0 Cacat | **Fuuton: Rasenshuriken!** | Shuriken angin raksasa berputar, memberikan **70 Damage**. |
| **OPEN PALM (Terbuka)** | $Defects \ge 3$, $S < 0.72$ | **Chakra Gatling!** | Tembakan beruntun 3 peluru energi, masing-masing **20 Damage**. |

---

## 5. Petunjuk Instalasi dan Eksekusi

### 5.1. Eksekusi Program
Jalankan melalui terminal:
```bash
python project-1/custom-minigame/main.py
```
Aplikasi akan membuka jendela mandiri berjudul **`Tugas Game - Shinobi Jutsu Battle`**.

### 5.2. Navigasi Tombol Pintas
* **`C`**: Buka mode kalibrasi warna kulit jika pencahayaan ruangan berubah.
* **`R`**: Reset pertandingan (*Restart Battle*) saat menang atau kalah.
* **`Q` / `ESC`**: Keluar dari permainan.
