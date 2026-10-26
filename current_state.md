# Status Terkini Proyek: PCV Project 1 (Praktikum Pengolahan Citra & Visi Komputer)

**Terakhir Diperbarui**: 27 Oktober 2026, 05:15 WIB  
**Repositori**: [`https://github.com/nicholasbenaya/PCV_Project-1`](https://github.com/nicholasbenaya/PCV_Project-1)  
**Cabang Aktif**: `main`  
**Lingkungan**: Python 3.12, OpenCV 5.0 / 4.x, NumPy, Pygame (Audio Sintetis)  

---

## 1. Ikhtisar Portofolio Proyek (3 Sub-Modul)

Repositori ini memuat 3 implementasi mandiri berbasis visi komputer murni (*No MediaPipe, No YOLO, No Deep Learning*):

1. **Tipe 1: Dota 2 Invoker Controller (`dota2-invoker/`)**  
   - Kontrol perapalan mantra hero Invoker langsung ke game Dota 2 via Windows API DirectInput (`SendInput`).
   - Klasifikasi gestur 1, 2, 3, 4, 5 jari terbuka, dan kepalan tangan menggunakan *convexity defects* & *bounding box geometry*.
2. **Tipe 2: Shinobi Jutsu Battle Mini-Game (`custom-minigame/`)**  
   - Game pertarungan sihir 2D murni OpenCV melawan *Demon Boss Susanoo*.
   - Klasifikasi 5 segel tangan (*Tiger, Snake, Cross, Ram, Open Palm*) berbasis *convexity defects*, rasio puncak 15%, dan *solidity*.
3. **Tipe 3: Whack-A-Mole Virtual Mini-Game (`whack-a-mole/`) [UPGRADED]**  
   - Game arcade memukul mole di atas canvas HD widescreen ($1280 \times 720$).
   - Dilengkapi sistem **Hand Pose Classification** (HIGH_FIVE move vs FIST whack), **Cursor Freeze Damping**, **Auto Face Blackout**, **PiP Diagnostik Fitur Bentuk PCV**, **3 Mode Segmentasi**, **Progression Level 1-10**, **Survival Mode**, dan **Audio Sintetis Adaptif**.

---

## 2. Rincian Teknis Whack-A-Mole (Tipe 3) Pasca Upgrade

### A. Sistem Klasifikasi 2 Hand Pose
- **Fitur Ekstraksi Citra**:
  - *Contour Area* ($A$) & *Convex Hull Area* ($A_{\text{hull}}$).
  - *Solidity*: $S = A / A_{\text{hull}}$.
  - *Bounding Box Aspect Ratio*: $AR = w / h$.
  - *Valid Convexity Defects*: Titik terdalam celah antar jari yang memenuhi kedalaman $d > 0.10 \times \min(w, h)$ dan sudut celah $\theta < 90^{\circ}$.
- **Aturan Klasifikasi**:
  - **`HIGH_FIVE` (Open Palm / Telapak Terbuka)**:
    - Terdeteksi jika $\text{Defects} \ge 3$ ATAU $S < 0.78$.
    - **Fungsi**: Membidik / mengarahkan kursor di layar. Tidak dapat memukul mole dan tidak dapat mengeklik tombol menu.
  - **`FIST` (Kepal / Clenched Hand)**:
    - Terdeteksi jika $\text{Defects} \le 1$ DAN $S > 0.83$ DAN $0.60 \le AR \le 1.40$.
    - **Fungsi**: Mengeksekusi aksi pukulan (**WHACK!**) pada mole atau mengeklik tombol menu yang sedang di-hover.
- **Temporal Smoothing Buffer**:
  - Menggunakan rolling buffer 5 frame dengan *majority voting* (`Counter.most_common`) untuk mencegah flickering saat tangan bergerak.

### B. Cursor Freeze / Damping Stabilizer
- Saat pemain mengepalkan tangan (`HIGH_FIVE` $\to$ `FIST`), kontur tangan menyusut dan titik centroid kerap bergeser ke bawah/belakang sebesar 20–60 piksel.
- Sistem membekukan (*freeze*) koordinat kursor ke posisi stabil telapak tangan terakhir selama jendela waktu $0.35\text{ detik}$, sehingga pukulan/klik tetap akurat mengenai lubang mole atau tombol yang dibidik.

### C. Ketahanan Segmentasi & Anti-Face Hijacking
- **Auto Face Blackout**:
  - Memindai wajah pemain menggunakan Haar Cascade (`haarcascade_frontalface_default.xml`).
  - Area wajah beserta margin leher ke bawah otomatis dinolkan pada masker biner ($M_{\text{face}} = 0$).
  - Dilengkapi *fallback* partisi spasial jika modul classifier tidak tersedia.
- **Dual-Section Partition**:
  - *Playing Section*: Area aktif target dan pelacakan tangan.
  - *Non-Playing Safe Zone*: Area kepala/tubuh pemain yang dinolkan secara fisik. Dapat ditukar orientasinya (Kanan/Kiri/Bawah) dengan tombol `[H]` atau `[TAB]`.
- **3 Preset Mode Segmentasi (Tombol `[M]`)**:
  1. `Skin (YCrCb + HSV)`: Segmentasi multi-ruang warna kulit manusia dengan adaptasi pencahayaan.
  2. `Black Glove`: Thresholding kontras tinggi untuk sarung tangan hitam pada latar terang.
  3. `High-Contrast Object`: Backup objek warna solid (misal bola/kain biru/merah) untuk kondisi ruangan darurat.
- **Kalibrasi Cepat (Tombol `[C]`)**:
  - Mengambil sampel $\mu$ dan $\sigma$ dari kotak target di layar, menghitung ulang batas lower/upper masker, dan menyimpannya permanen di `save_data.json`.

### D. Jendela PiP Diagnostik Fitur Bentuk PCV (Tombol `[V]`)
- Menampilkan jendela Picture-in-Picture di pojok layar beresolusi $280 \times 210\text{ px}$ yang memvisualisasikan:
  1. Masker biner segmentasi terkini.
  2. Garis kontur tangan asli (kuning).
  3. Garis poligon **Convex Hull** (hijau neon).
  4. Titik **Convexity Defects** (lingkaran merah/kuning).
  5. Metrik nilai: *Pose Status, Defect Count, Solidity, Aspect Ratio, dan Mode Aktif*.
  - Sangat krusial untuk diperlihatkan ke dosen/asisten penguji saat sesi demo pembuktian algoritma PCV.

---

## 3. Matriks Berkas Modul Whack-A-Mole

| Berkas | Peran dan Tanggung Jawab |
| :--- | :--- |
| [`config.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/config.py) | Definisi resolusi HD 1280x720, batas HSV/YCrCb, konstanta pose (HIGH_FIVE/FIST), 10 level, 3 preset mode. |
| [`hand_tracker.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/hand_tracker.py) | Segmentasi, Auto Face Blackout, ekstraksi Convex Hull & Defects, Solidity, Pose Voting, dan Cursor Freeze. |
| [`entities.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/entities.py) | Entitas Hole, Mole, HitEffect, ScoreManager, SurvivalTracker, dan InteractiveButton dengan FIST trigger. |
| [`game_engine.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/game_engine.py) | State Machine (Menu, Level Select, Grid Select, Kalibrasi, Survival, Level Mode, GameOver), validasi hit FIST. |
| [`ui_renderer.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/ui_renderer.py) | Kursor dinamis (Hijau MOVE vs Merah WHACK), PiP Diagnostik Fitur Bentuk PCV, HUD survival/level, kartu menu. |
| [`audio_system.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/audio_system.py) | Generator audio sintetis NumPy stereo (click, bonk, miss, clear, game over) dan loop BGM 4-tier. |
| [`save_manager.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/save_manager.py) | Persistensi data JSON (level unlock 1-10, skor survival, kalibrasi warna, grid terpilih). |
| [`main.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/main.py) | Entry point game loop, penanganan hotkey (`C`, `M`, `V`, `H`, `S`, `ESC`), dan integrasi perangkat webcam. |

---

## 4. Panduan Kontrol & Demonstrasi

### Kontrol Gestur Tangan:
- **`OPEN PALM (HIGH_FIVE)`**: Arahkan kursor tangan ke lubang mole atau tombol menu. Kursor berwarna hijau dengan label `"MOVE (OPEN)"`.
- **`FIST (KEPAL)`**: Kepalkan tangan tepat di atas target untuk memukul (**WHACK!**) atau memilih tombol menu. Kursor berubah menjadi merah pekat dengan efek impak dan label `"WHACK! (FIST)"`.

### Kontrol Keyboard Lengkap:
| Tombol | Fungsi |
| :--- | :--- |
| **`V` / `P`** | Toggle jendela PiP Diagnostik Fitur Bentuk PCV (Convex Hull, Defek, Solidity, Pose) |
| **`M`** | Berganti Preset Segmentasi (`Skin` $\to$ `Black Glove` $\to$ `High-Contrast Object`) |
| **`C`** | Masuk Mode Kalibrasi Cepat / Ambil sampel warna kulit |
| **`H` / `TAB`** | Ganti Sisi Playing Section (Kanan, Kiri, Bawah) |
| **`S`** | Menyalakan / mematikan suara (*Mute Audio*) |
| **`1` - `6`** | Memilih menu utama via keyboard |
| **`1` - `0`** | Memilih level 1 s.d. 10 pada layar Level Select |
| **`1` - `4`** | Memilih ukuran grid pada layar Grid Select |
| **`R`** | Restart sesi permainan saat Game Over / Level Cleared |
| **`SPASI`** | Kunci kalibrasi / Lanjut ke level berikutnya saat Level Cleared |
| **`ESC`** | Kembali ke Menu Utama dari layar mana pun / Keluar permainan |

---

## 5. Status Verifikasi Pengujian

- [x] Deteksi cacat kecembungan (*Convexity Defects*) terverifikasi akurat menghitung celah antar jari.
- [x] Klasifikasi pose `HIGH_FIVE` dan `FIST` terverifikasi konvergen stabil via voting buffer 5 frame.
- [x] Mekanisme *Cursor Freeze Damping* terverifikasi mencegah pergeseran kursor saat kepalan tangan dibuat.
- [x] *Auto Face Blackout* terintegrasi menghapus derau wajah dan leher dari masker biner.
- [x] Mode interaksi: `HIGH_FIVE` membidik tanpa memukul, `FIST` berhasil memicu tabrakan mole dan klik menu.
- [x] Jendela PiP Diagnostik PCV merender poligon Convex Hull dan titik defek secara real-time tanpa *frame-rate drop*.
- [x] Seluruh alur Level 1-10, Mode Survival, dan Audio Prosedural berjalan normal tanpa regresi.
