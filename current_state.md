# Status Terkini Proyek: PCV Project 1 (Praktikum Pengolahan Citra & Visi Komputer)

**Terakhir Diperbarui**: 27 Oktober 2026, 07:35 WIB  
**Repositori**: [`https://github.com/nicholasbenaya/PCV_Project-1`](https://github.com/nicholasbenaya/PCV_Project-1)  
**Cabang Aktif**: `safe-area` (Branch Baru Khusus Partisi Safe Area & Dynamic Whack Timing)  
**Lingkungan**: Python 3.12, OpenCV 5.0 / 4.x, NumPy, Pygame (Audio Sintetis)  

---

## 1. Ikhtisar Portofolio Proyek (3 Sub-Modul)

Repositori ini memuat 3 implementasi mandiri berbasis visi komputer murni (**100% Bebas Model Pembelajaran / Tanpa MediaPipe / Tanpa DNN / Tanpa Haar Cascade XML**):

1. **Tipe 1: Dota 2 Invoker Controller (`dota2-invoker/`)**  
   - Kontrol perapalan mantra hero Invoker langsung ke game Dota 2 via Windows API DirectInput (`SendInput`).
   - Klasifikasi gestur 1, 2, 3, 4, 5 jari terbuka, dan kepalan tangan menggunakan *convexity defects* & *bounding box geometry*.
2. **Tipe 2: Shinobi Jutsu Battle Mini-Game (`custom-minigame/`)**  
   - Game pertarungan sihir 2D murni OpenCV melawan *Demon Boss Susanoo*.
   - Klasifikasi 5 segel tangan (*Tiger, Snake, Cross, Ram, Open Palm*) berbasis *convexity defects*, rasio puncak 15%, dan *solidity*.
3. **Tipe 3: Whack-A-Mole Virtual Mini-Game (`whack-a-mole/`) [DUAL-SECTION SAFE AREA RESTORED]**  
   - Game arcade memukul mole di atas canvas HD ($1280 \times 720$) dengan partisi **Dual-Section Safe Area** yang memblokir 100% wajah, tubuh, dan rambut pemain.
   - Mengintegrasikan **4 Pilar PCD Klasik & Safe Area**:
     1. **Partisi Dual-Section Safe Area (Tombol `[H]` / `[TAB]`)**: Menolkan seluruh masker biner di luar Playing Section.
     2. **Background Subtraction Berbasis Referensi (Tombol `[B]`)** via `cv2.absdiff()`.
     3. **Eliminasi Wajah via Topologi Hierarki Kontur (`cv2.RETR_CCOMP`)** berbasis lubang internal mata/mulut.
     4. **Mekanisme "Open-Palm Lock-On" & Nearest-Neighbor Spatial Tracking** dengan Area Confidence Scoring & Filter Dimensi Minimal ($55\text{ px}$).
     5. **Sistem Klasifikasi 2 Hand Pose (HIGH_FIVE move vs FIST whack)** dengan **Cursor Position Freeze Damping**.
   - Dilengkapi jendela **PiP Diagnostik Fitur Bentuk PCV (Tombol `[V]`)**, **Progression Level 1-10**, **Survival Mode**, dan **Audio Sintetis Adaptif**.

---

## 2. Rincian Teknis Whack-A-Mole (Tipe 3) pada Branch `safe-area`

### A. Fitur Utama: Partisi Dual-Section Safe Area (Tombol `[H]` / `[TAB]`)
- **Tujuan**: Mengeliminasi 100% gangguan wajah, kepala, rambut, bahu, dan bayangan tubuh pemain tanpa mengandalkan posisi duduk yang kaku.
- **Mekanisme Masking Fisik**:
  Pada [`hand_tracker.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/hand_tracker.py), jika `play_rect` aktif:
  ```python
  isolated_mask = np.zeros_like(raw_mask)
  isolated_mask[py1:py2, px1:px2] = raw_mask[py1:py2, px1:px2]
  raw_mask = isolated_mask
  ```
  Seluruh piksel di luar Playing Section bernilai 0 (hitam total), sehingga tidak ada kontur wajah/rambut yang dapat terdeteksi oleh OpenCV.
- **4 Pilihan Preset Zona (Siklus via Hotkey `[H]` atau `[TAB]`)**:
  1. **`TANGAN KANAN (Kiri: Wajah Diblokir)`** (Default): `play_rect = (420, 0, 1280, 720)`, `safe_rect = (0, 0, 420, 720)`. Sisi kiri tempat pemain duduk diblokir total; tangan kanan bebas bermain di sisi kanan.
  2. **`TANGAN KIRI (Kanan: Wajah Diblokir)`**: `play_rect = (0, 0, 860, 720)`, `safe_rect = (860, 0, 1280, 720)`.
  3. **`TANGAN BAWAH (Atas: Wajah Diblokir)`**: `play_rect = (0, 240, 1280, 720)`, `safe_rect = (0, 0, 1280, 240)`.
  4. **`LAYAR PENUH (Tanpa Safe Zone)`**: `play_rect = (0, 0, 1280, 720)` untuk demonstrasi deteksi layar penuh murni.
- **Visualisasi Safe Area**:
  - Area Non-Playing Safe Zone diredupkan ($40\%$ kecerahan).
  - Garis pemisah neon cyan tebal dengan label: `"NON-PLAYING SECTION [SAFE ZONE] Wajah, Rambut & Tubuh Terblokir 100%"`.
  - Grid lubang, tombol interaktif, HUD, dan kotak dialog modal secara otomatis menyesuaikan tata letak di dalam `play_rect`.

### B. Pilar 1: Background Subtraction Multi-Channel & Invalidation (Tombol `[B]` & `[N]`)
- Pemain merekam frame referensi latar belakang saat duduk di depan webcam dengan menekan tombol `[B]`.
- Menggunakan selisih multi-channel BGR + Chroma YCrCb (kebal metamerisme grayscale) dengan threshold adaptif $T_{\text{bg}} = 18$.
- Menerapkan **Static Background Artifact Invalidation**: Hanya area yang *sudah ada dan sewarna kulit di background* (wajah/meja kayu yang diam) yang dieliminasi, sehingga tangan yang diangkat di depan dinding/baju tidak akan pernah ikut tersubtrak/hilang.
- Tombol **`[N]`** disediakan untuk mereset / me-bypass background subtraction secara instan.
- **Pemisahan Frame Rendering (`display_frame` vs `raw_frame`)**: Memisahkan kanvas rendering UI dari frame kamera mentah, menjamin tombol menu, HUD teks, dan PiP tidak akan pernah ikut terekam ke dalam frame referensi background saat pemain menekan `[B]`, mencegah tangan teriris/terpotong di belakang tombol menu.
- **Khusus Black Glove Mode (`PRESET_GLOVE`)**:
  1. *Unified Multi-Channel Background Invalidation (`[B]`)*: Rambut hitam dan pakaian gelap yang diam terekam saat tombol `[B]`, sehingga ketika tangan bersarung bergerak, hanya sarung tangan yang terdeteksi tanpa ada kotak hitam artifisial.
  2. *Pose-Scoring Nearest Neighbor*: Memberikan penalti jarak (+80 px) pada kontur amorf tanpa pose (seperti sisa rambut) sehingga kursor selalu mengunci sarung tangan dan tidak tersangkut di rambut saat tangan melintas di dekat kepala.

### B. Pilar 2: Eliminasi Wajah via Topologi Hierarki Kontur (`cv2.RETR_CCOMP`)
- Tanpa model learning / XML classifier, sistem membedakan wajah dari tangan menggunakan analisis lubang internal kontur.
- Kontur dicari dengan mode `cv2.RETR_CCOMP` pada masker sebelum morphological closing besar.
- Menghitung jumlah child contour internal dengan luas $\ge 25\text{ px}$. Karena wajah memiliki mata, alis, dan mulut yang menghasilkan $\ge 3$ lubang internal, kontur dengan $\ge 3$ lubang dan defek celah jari $< 2$ **otomatis ditolak sebagai wajah**.

### C. Pilar 3: Open-Palm Lock-On, Area Confidence, & Sticky Object Tracking
- **Filter Dimensi Minimal (`min-width`, `min-height`, `min-area`)**:
  - Ambang batas kontur tangan ditingkatkan menjadi $A \ge 2500\text{ px}$, lebar bounding box $w \ge 55\text{ px}$, dan tinggi $h \ge 55\text{ px}$.
  - Secara efektif mengeliminasi serpihan bising kecil, garis rambut tipis, atau bayangan tepi meja agar kursor tidak terlempar.
- **State Belum Terkunci (`is_hand_locked == False`)**:
  - Sistem mengurutkan kandidat kontur berdasarkan luas terbesar (`area` descending).
  - Sistem mencari kontur telapak tangan terbuka (*HIGH_FIVE*) dengan syarat:
    $$\text{Valid Defects} \ge 1 \quad (\text{minimal 1 titik merah sela jari}) \quad \lor \quad \text{Solidity} < 0.80$$
  - Begitu ditemukan, sistem mengaktifkan lock-on, mencatat luas awal $A_0 = \text{area}$, dan mengunci objek tersebut.
- **State Terkunci (`is_hand_locked == True`) - Area Confidence & Sticky Locking**:
  - Sistem mengunci ketat hanya pada objek tangan tersebut dalam radius $\le 180\text{ px}$ menggunakan scoring **Area Confidence & Consistency**:
    $$\text{Total Cost} = d + \left|1.0 - \min\left(2.5, \frac{A}{A_{\text{last}}}\right)\right| \cdot 60.0 + \text{Penalti Serpihan} - \left(\frac{A}{A_{\text{max}}}\right) \cdot 40.0 + \text{Penalti Pose}$$
  - Serpihan kecil ($A < 0.5 \times A_{\text{last}}$) diganjar penalti jarak $+90\text{ px}$, sehingga kontur tangan utama selalu mengalahkan serpihan bising yang kebetulan berjarak dekat.
  - Luas tangan diperbarui secara adaptif menggunakan EMA: $A_{\text{last}} = 0.85 \times A_{\text{last}} + 0.15 \times A_{\text{current}}$.
  - Jika tangan tertutup sejenak (occlusion 1-2 frame), sistem menahan kursor pada koordinat terakhir tangan dan tidak melompat.
  - Kunci hanya dilepas jika tangan benar-benar ditarik keluar dari layar selama $> 24\text{ frame}$ (~0.8 detik).

### D. Pilar 4: Ekstraksi Fitur 2 Pose Sederhana (HIGH_FIVE vs FIST) & Eliminasi Titik Merah Palsu
- **Akar Masalah Titik Merah Kiri Atas pada Kepalan**:
  - Pada tangan kanan yang mengepal ke arah webcam, lekukan antara jempol terlipat dan buku jari telunjuk menghasilkan ceruk dengan kedalaman $\approx 8 - 12\text{ px}$.
  - Dengan parameter lama `DEFECT_MIN_DEPTH_RATIO = 0.08`, ceruk jempol ini lolos seleksi defek ($100\text{ px} \times 0.08 = 8\text{ px}$) dan ditandai lingkaran merah di sebelah kiri atas kepalan tangan.
  - Akibat `HIGH_FIVE_MIN_DEFECTS = 1`, kepalan tangan penuh tersebut keliru terbaca sebagai `HIGH_FIVE` dan gagal mengeksekusi pukulan (`WHACK!`).
- **Solusi Rekayasa Geometri PCD Klasik**:
  1. *Ambang Kedalaman Proporsional & Minimal Absolut*: Menaikkan `DEFECT_MIN_DEPTH_RATIO = 0.15` dan menetapkan `DEFECT_MIN_PIXEL_DEPTH = 14.0 px`. Celah sela jari asli pada telapak terbuka memiliki kedalaman $25 - 60\text{ px}$ sehingga tetap terdeteksi mulus, sedangkan lekukan jempol kepalan ($< 12\text{ px}$) tereliminasi total (nol titik merah).
  2. *Filter Pergelangan Tangan & Garis Batas*: Mengabaikan defek di 18% terbawah bounding box (area masuk pergelangan tangan) dan defek yang menempel pada garis batas vertikal Safe Area.
  3. *Klasifikasi Pose Berbasis Konsistensi Morfologi*:
     - `HIGH_FIVE` (Move / Hover): Terdeteksi jika $\text{Defects} \ge 2$, ATAU $\text{Defects} \ge 1$ dengan kontur renggang ($\text{Solidity} < 0.78$).
     - `FIST` (Whack / Click): Kepalan padat cembung ($\text{Solidity} \ge 0.78$ dan $\text{Defects} \le 1$), menjamin kepalan tangan penuh 100% konsisten terklasifikasi sebagai `FIST`.
- **Cursor Freeze Damping**: Saat transisi ke FIST, kursor dibekukan di koordinat *HIGH_FIVE* terakhir selama $0.35\text{ detik}$ agar pukulan tidak meleset.

### E. Jendela PiP Diagnostik Fitur Bentuk PCV (Tombol `[V]`)
- Menampilkan jendela Picture-in-Picture di pojok layar beresolusi $280 \times 210\text{ px}$ yang memvisualisasikan:
  1. Masker biner segmentasi aktif.
  2. Garis kontur tangan asli (kuning).
  3. Garis poligon **Convex Hull** (hijau tebal).
  4. Titik **Convexity Defects** (lingkaran merah/kuning).
  5. Metrik nilai: *Lock Status, BG Sub Status, Defect Count, Internal Holes, Solidity, Aspect Ratio, dan Status Pose*.

### F. Perbaikan Waktu Game Over, Restart Mode Level, & Durasi Proporsional Mole per Whack
1. **Perbaikan Reset Waktu & Restart Akurat Level yang Sama Saat Game Over**:
   - Membekukan waktu bertahan (`final_survival_elapsed`) saat Game Over terjadi, sehingga waktu di layar Game Over tidak lagi terus bertambah.
   - **Fix Restart Level Mode**: Memastikan `self.last_played_mode = "LEVEL"` dan `self.survival_tracker = None` secara tegas saat Game Over di Mode Level. Saat pemain menekan tombol `[SPASI]`, `[ENTER]`, atau `[R]`, sistem memanggil `restart_current_game()` yang secara konsisten me-restart **level yang sama persis** (misal: Level 3 tetap Level 3) dengan countdown timer kembali penuh, skor di-reset ke 0, dan tidak tertukar ke mode Survival.
   - Hotkey `[SPASI]`, `[ENTER]`, dan `[R]` kini sepenuhnya didukung di layar Game Over.
2. **Durasi Kemunculan Mole Lebih Lama & Adaptif Proporsional**:
   - **Mode Level Progression (Level 1-10)**:
     - Durasi kemunculan mole dinaikkan (+50%) menjadi $2.70\text{s} - 3.20\text{s}$ di Level 1, menurun secara bertahap dan proporsional hingga $1.25\text{s} - 1.65\text{s}$ di Level 10.
     - Setiap pukulan sukses (**WHACK!**) memberikan **Bonus Waktu Tambahan** pada countdown timer yang proporsional terhadap level (Level 1: $+1.5\text{s}$, menurun bertahap hingga $+0.6\text{s}$ di Level 10).
     - Efek visual `HitEffect` menampilkan teks bonus: `WHACK! +100 (+1.5s)`.
   - **Mode Survival**:
     - Durasi kemunculan mole dinaikkan menjadi $2.70\text{s} - 3.20\text{s}$ di awal permainan, menurun secara proporsional dan bertahap seiring berjalannya waktu bertahan (hingga $1.30\text{s} - 1.70\text{s}$ pada $t \ge 150\text{ detik}$).
     - Setiap kombo kelipatan 8 memberikan regenerasi $+1$ HP (nyawa).

---

## 3. Matriks Berkas Modul Whack-A-Mole

| Berkas | Peran dan Tanggung Jawab |
| :--- | :--- |
| [`config.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/config.py) | Parameter Full Screen HD 1280x720, threshold defek (<85 deg), lock-on radius (160 px), 3 preset mode, 10 level. |
| [`hand_tracker.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/hand_tracker.py) | Implementasi 4 Pilar PCD: BG Subtraction absdiff, RETR_CCOMP face hole rejection, Open-Palm Lock-On, Nearest-Neighbor, dan Cursor Freeze. |
| [`entities.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/entities.py) | Entitas Hole, Mole, HitEffect, ScoreManager, SurvivalTracker, dan InteractiveButton dengan FIST trigger. |
| [`game_engine.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/game_engine.py) | State Machine (Menu, Level Select, Grid Select, Kalibrasi, Survival, Level Mode, GameOver), validasi hit FIST pada layar penuh. |
| [`ui_renderer.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/ui_renderer.py) | Full Screen HUD, kursor dinamis (Hijau MOVE vs Merah WHACK), PiP Diagnostik Fitur Bentuk PCV, banner lock-on. |
| [`audio_system.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/audio_system.py) | Generator audio sintetis NumPy stereo (click, bonk, miss, clear, game over) dan loop BGM 4-tier. |
| [`save_manager.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/save_manager.py) | Persistensi data JSON (level unlock 1-10, skor survival, kalibrasi warna, grid terpilih). |
| [`main.py`](file:///c:/Users/kegz/Desktop/Folder%20iko/Projects%20Local/PCV/project-1/whack-a-mole/main.py) | Entry point game loop, penanganan hotkey (`B`, `V`, `M`, `C`, `S`, `ESC`), dan integrasi perangkat webcam. |

---

## 4. Panduan Kontrol & Demonstrasi

### Kontrol Gestur Tangan:
- **`OPEN PALM (HIGH_FIVE)`**: Arahkan kursor tangan ke lubang mole atau tombol menu. Kursor berwarna hijau dengan label `"MOVE (OPEN)"`.
- **`FIST (KEPAL)`**: Kepalkan tangan tepat di atas target untuk memukul (**WHACK!**) atau memilih tombol menu. Kursor berubah menjadi merah pekat dengan efek impak dan label `"WHACK! (FIST)"`.

### Kontrol Keyboard Lengkap:
| Tombol | Fungsi |
| :--- | :--- |
| **`H` / `TAB`** | **Ganti Orientasi Safe Zone** (`Tangan Kanan` $\to$ `Tangan Kiri` $\to$ `Tangan Bawah` $\to$ `Layar Penuh`) |
| **`B`** | **Merekam Frame Referensi Latar Belakang** (*Background Subtraction*) |
| **`N`** | **Reset / Bypass Referensi Latar Belakang** |
| **`V` / `P`** | Toggle jendela **PiP Diagnostik Fitur Bentuk PCV** (Hull, Defek, Solidity, Pose) |
| **`M`** | Berganti Preset Segmentasi (`Skin` $\to$ `Black Glove` $\to$ `High-Contrast Object`) |
| **`C`** | Masuk Mode Kalibrasi Cepat / Ambil sampel warna kulit |
| **`S`** | Menyalakan / mematikan audio sintetis (*Mute Audio*) |
| **`1` - `6`** | Memilih menu utama via keyboard |
| **`1` - `0`** | Memilih level 1 s.d. 10 pada layar Level Select |
| **`1` - `4`** | Memilih ukuran grid pada layar Grid Select |
| **`R` / `SPASI` / `ENTER`** | Restart sesi permainan saat Game Over (mengulang mode/level yang sama persis) |
| **`SPASI`** | Lanjut ke level berikutnya saat Level Cleared |
| **`M` / `ESC`** | Kembali ke Menu Utama dari layar mana pun / Keluar permainan |

---

## 5. Status Verifikasi Pengujian

- [x] **Zero Learning Models**: Terverifikasi 100% bebas dari MediaPipe, DNN, dan Haar Cascade XML.
- [x] **Zero Safe Area / No Static Partitions**: Tangan terverifikasi bebas bergerak di seluruh resolusi $1280 \times 720$.
- [x] **Background Subtraction (`cv2.absdiff`)**: Terverifikasi memisahkan foreground gerak dari latar belakang statis.
- [x] **Hierarki Kontur (`cv2.RETR_CCOMP`)**: Terverifikasi mendeteksi $\ge 3$ lubang internal pada wajah dan menolaknya.
- [x] **Open-Palm Lock-On**: Terverifikasi hanya mengunci saat telapak tangan terbuka (defek $\ge 3$, sudut $< 85^\circ$, solidity $\le 0.82$). Wajah terbukti tidak dapat memicu lock-on.
- [x] **Nearest-Neighbor Spatial Tracking**: Terverifikasi melacak tangan saat transisi ke FIST ($d \le 160\text{ px}$) tanpa melompat ke objek lain.
- [x] **Gameplay & Audio**: Mode Survival, Mode Level 1-10, Grid multi-size, dan audio prosedural terverifikasi 100% berjalan normal.
