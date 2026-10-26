# Status Terkini Proyek: PCV Project 1 (Praktikum Pengolahan Citra & Visi Komputer)

**Terakhir Diperbarui**: 27 Oktober 2026, 05:45 WIB  
**Repositori**: [`https://github.com/nicholasbenaya/PCV_Project-1`](https://github.com/nicholasbenaya/PCV_Project-1)  
**Cabang Aktif**: `main`  
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
3. **Tipe 3: Whack-A-Mole Virtual Mini-Game (`whack-a-mole/`) [UPGRADED 4 PILAR PCD KLASIK]**  
   - Game arcade memukul mole di atas canvas layar penuh HD ($1280 \times 720$) tanpa batasan Safe Area / partisi statis.
   - Mengintegrasikan **4 Pilar PCD Klasik**:
     1. **Background Subtraction Berbasis Referensi (Tombol `[B]`)** via `cv2.absdiff()`.
     2. **Eliminasi Wajah via Topologi Hierarki Kontur (`cv2.RETR_CCOMP`)** berbasis lubang internal mata/mulut.
     3. **Mekanisme "Open-Palm Lock-On" & Nearest-Neighbor Spatial Tracking** (mencegah pembajakan kursor oleh kepala atau objek lain).
     4. **Sistem Klasifikasi 2 Hand Pose (HIGH_FIVE move vs FIST whack)** dengan **Cursor Position Freeze Damping**.
   - Dilengkapi jendela **PiP Diagnostik Fitur Bentuk PCV (Tombol `[V]`)**, **Progression Level 1-10**, **Survival Mode**, dan **Audio Sintetis Adaptif**.

---

## 2. Rincian Teknis Whack-A-Mole (Tipe 3) Pasca Upgrade 4 Pilar PCD

### A. Pilar 1: Background Subtraction Multi-Channel & Invalidation (Tombol `[B]` & `[N]`)
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

### C. Pilar 3: Open-Palm Lock-On & Sticky Object Tracking
- **State Belum Terkunci (`is_hand_locked == False`)**:
  - Sistem mencari kontur telapak tangan terbuka (*HIGH_FIVE*) dengan syarat:
    $$\text{Valid Defects} \ge 1 \quad (\text{minimal 1 titik merah sela jari}) \quad \lor \quad \text{Solidity} < 0.80$$
  - Begitu ditemukan, sistem mengaktifkan lock-on dan mengunci objek tersebut.
- **State Terkunci (`is_hand_locked == True`) - Sticky Locking**:
  - Sistem mengunci ketat hanya pada objek tangan tersebut dalam radius $\le 180\text{ px}$.
  - Tidak akan pernah melompat ke objek lain (rambut, wajah, latar belakang). Kontur tanpa pose diberi penalti jarak (+80 px).
  - Jika tangan tertutup sejenak (occlusion 1-2 frame), sistem menahan kursor pada koordinat terakhir tangan dan tidak melompat.
  - Kunci hanya dilepas jika tangan benar-benar ditarik keluar dari layar selama $> 24\text{ frame}$ (~0.8 detik).

### D. Pilar 4: Ekstraksi Fitur 2 Pose Sederhana (HIGH_FIVE vs FIST)
- **`HIGH_FIVE` (Move / Open Palm)**: Terdeteksi jika ada **minimal 1 titik merah sela jari** ($\text{Defects} \ge 1 \lor \text{Solidity} < 0.80$). Berfungsi menggerakkan kursor.
- **`FIST` (Whack / Click)**: Sisanya, yaitu saat tidak ada titik merah sela jari ($\text{Defects} == 0 \land \text{Solidity} \ge 0.80$, kepalan tangan tertutup rapat). Berfungsi memukul mole atau memilih menu.
- **Cursor Freeze Damping**: Saat transisi ke FIST, kursor dibekukan di koordinat *HIGH_FIVE* terakhir selama $0.35\text{ detik}$ agar pukulan tidak meleset.

### E. Jendela PiP Diagnostik Fitur Bentuk PCV (Tombol `[V]`)
- Menampilkan jendela Picture-in-Picture di pojok layar beresolusi $280 \times 210\text{ px}$ yang memvisualisasikan:
  1. Masker biner segmentasi aktif.
  2. Garis kontur tangan asli (kuning).
  3. Garis poligon **Convex Hull** (hijau tebal).
  4. Titik **Convexity Defects** (lingkaran merah/kuning).
  5. Metrik nilai: *Lock Status, BG Sub Status, Defect Count, Internal Holes, Solidity, Aspect Ratio, dan Status Pose*.

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
| **`B`** | **Merekam Frame Referensi Latar Belakang** (*Background Subtraction*) |
| **`V` / `P`** | Toggle jendela **PiP Diagnostik Fitur Bentuk PCV** (Hull, Defek, Solidity, Pose) |
| **`M`** | Berganti Preset Segmentasi (`Skin` $\to$ `Black Glove` $\to$ `High-Contrast Object`) |
| **`C`** | Masuk Mode Kalibrasi Cepat / Ambil sampel warna kulit |
| **`S`** | Menyalakan / mematikan audio sintetis (*Mute Audio*) |
| **`1` - `6`** | Memilih menu utama via keyboard |
| **`1` - `0`** | Memilih level 1 s.d. 10 pada layar Level Select |
| **`1` - `4`** | Memilih ukuran grid pada layar Grid Select |
| **`R`** | Restart sesi permainan saat Game Over / Level Cleared |
| **`SPASI`** | Lanjut ke level berikutnya saat Level Cleared |
| **`ESC`** | Kembali ke Menu Utama dari layar mana pun / Keluar permainan |

---

## 5. Status Verifikasi Pengujian

- [x] **Zero Learning Models**: Terverifikasi 100% bebas dari MediaPipe, DNN, dan Haar Cascade XML.
- [x] **Zero Safe Area / No Static Partitions**: Tangan terverifikasi bebas bergerak di seluruh resolusi $1280 \times 720$.
- [x] **Background Subtraction (`cv2.absdiff`)**: Terverifikasi memisahkan foreground gerak dari latar belakang statis.
- [x] **Hierarki Kontur (`cv2.RETR_CCOMP`)**: Terverifikasi mendeteksi $\ge 3$ lubang internal pada wajah dan menolaknya.
- [x] **Open-Palm Lock-On**: Terverifikasi hanya mengunci saat telapak tangan terbuka (defek $\ge 3$, sudut $< 85^\circ$, solidity $\le 0.82$). Wajah terbukti tidak dapat memicu lock-on.
- [x] **Nearest-Neighbor Spatial Tracking**: Terverifikasi melacak tangan saat transisi ke FIST ($d \le 160\text{ px}$) tanpa melompat ke objek lain.
- [x] **Gameplay & Audio**: Mode Survival, Mode Level 1-10, Grid multi-size, dan audio prosedural terverifikasi 100% berjalan normal.
