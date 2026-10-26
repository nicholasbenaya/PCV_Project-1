# Laporan Teknis Proyek 1 - Tipe 3: Whack-A-Mole Virtual Mini-Game (Pure OpenCV)

Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)  
Topik: *Virtual Interactive Whack-A-Mole Game with Adaptive Audio, Multi-Grid, Survival & Level Progression*  
Berkas Utama: [`main.py`](main.py) | [`game_engine.py`](game_engine.py) | [`hand_tracker.py`](hand_tracker.py) | [`audio_system.py`](audio_system.py) | [`ui_renderer.py`](ui_renderer.py) | [`entities.py`](entities.py) | [`save_manager.py`](save_manager.py) | [`config.py`](config.py)

---

## 1. Pendahuluan

**Whack-A-Mole Virtual** adalah game arcade interaktif berbasis *Computer Vision* murni (**OpenCV `cv2` & NumPy**) tanpa pustaka *deep learning* (tanpa MediaPipe, YOLO, TensorFlow, atau PyTorch). Pemain mengontrol seluruh navigasi menu dan aksi pemukulan mole secara *hands-free* di depan webcam melalui pelacakan titik berat (*centroid*) telapak tangan.

Proyek ini dilengkapi dengan:
1. **Resolusi HD Widescreen ($1280 \times 720$)**: Tampilan lega dan responsif untuk kenyamanan visual serta fleksibilitas jangkauan gerak tangan pemain.
2. **Dual-Section Partition (Anti-Face Hijacking)**: Pembagian layar menjadi dua zona:
   - **Playing Section**: Area aktif tempat target mole, kursor tangan, dan tombol menu berada.
   - **Non-Playing Safe Zone**: Area tempat kepala/tubuh pemain berada yang secara fisik dinolkan pada masker biner segmentasi kulit ($M_{\text{isolated}} = 0$). Hal ini memberikan jaminan 100% wajah tidak akan pernah membajak kursor tangan.
   - **Preset Fleksibel**: Tombol `[H]` atau `[TAB]` untuk berganti orientasi: Tangan Kanan (default), Tangan Kiri (*left-handed*), atau Tangan Bawah.
3. **Mode Kalibrasi Warna Kulit Interaktif**: Mengambil sampel statistik telapak tangan pemain pada ruang warna YCrCb dan HSV untuk beradaptasi terhadap berbagai rona kulit dan kondisi pencahayaan ruangan secara presisi.
4. **Main Menu Lengkap & Navigasi Fleksibel**:
   - Pilihan konfigurasi Grid ($2 \times 2$, $2 \times 3$, $3 \times 3$, $3 \times 4$).
   - Pilihan Mode Survival (Endless High Score dengan intensitas meningkat) & Mode Level (Level 1 - 10 dengan sistem *unlock*).
   - Navigasi tombol dapat menggunakan keyboard angka `[1-6]` atau **Virtual Touch** (arahkan kursor tangan di atas tombol selama ~0.85 detik).
5. **Sintesis Audio Prosedural & BGM Dinamis Adaptif**: Seluruh efek suara (klik tombol, pukulan *bonk*, mole kabur, *level clear fanfare*, *game over*) serta 4 tingkat loop musik latar *chiptune* disintesis secara matematis menggunakan rumus gelombang NumPy tanpa file audio eksternal. Musik in-game beradaptasi semakin cepat dan bersemangat seiring meningkatnya intensitas permainan.

---

## 2. Arsitektur Modular & Desain Sistem

Kode dibangun dengan arsitektur bersih (*Clean Code & Separation of Concerns*):

```text
whack-a-mole/
│
├── config.py           # Parameter konstan: resolusi HD 1280x720, batas warna HSV/YCrCb, preset zona, 10 level
├── audio_system.py     # Generator gelombang audio sintetis (NumPy stereo PCM) & BGM loop adaptif
├── save_manager.py     # Persistensi data progres: level unlock, rekor survival, kalibrasi, preferensi grid
├── hand_tracker.py     # Modul Vision: isolasi Playing Section, sampling kalibrasi, morfologi & centroid EMA
├── entities.py         # Model objek: Hole, Mole (multi-mole), HitEffect, ScoreManager, SurvivalTracker, Button
├── ui_renderer.py      # Modul visualisasi OpenCV: overlay dual-section, HUD, Mole kartun, modal & PiP
├── game_engine.py      # Mesin utama: State Machine (MENU, LEVEL_SELECT, GRID_SELECT, CALIBRATION, PLAYING, GAMEOVER)
├── main.py             # Entry point aplikasi: loop kamera, event hotkey, kalibrasi, dan eksekusi
├── save_data.json      # File penyimpanan lokal progres pemain (dibuat otomatis)
└── README.md           # Laporan teknis dan dokumentasi lengkap
```

---

## 3. Metodologi Pengolahan Citra Digital (PCD)

### 3.1. Isolasi Zona Bermain (*Playing Section Masking*)
Penyebab utama wajah membajak kursor pada segmentasi kulit sederhana adalah luas area wajah/leher yang lebih besar daripada tangan. Sistem ini menerapkan partisi spasial tegas:
$$\mathbf{M}_{\text{isolated}}(x, y) = \begin{cases} \mathbf{M}_{\text{skin}}(x, y) & \text{jika } (x, y) \in \text{Playing Section} \\ 0 & \text{jika } (x, y) \in \text{Non-Playing Safe Zone} \end{cases}$$
Dengan teknik ini, seluruh piksel di Non-Playing Safe Zone bernilai $0$, sehingga operasi penemuan kontur (`cv2.findContours`) hanya memproses piksel di Playing Section. Wajah pemain 100% tereliminasi secara fisik dari komputasi.

### 3.2. Segmentasi Multi-Ruang Warna (YCrCb + HSV)
Memisahkan intensitas pencahayaan dari nilai krominans:
* **Ruang YCrCb (Invarian Pencahayaan):**
  $$Y \in [0, 255], \quad Cr \in [133, 175], \quad Cb \in [77, 127]$$
* **Ruang HSV (Rona & Kejenuhan Kulit):**
  $$H \in [0, 25] \cup [170, 180], \quad S \in [30, 210], \quad V \in [50, 255]$$
* **Penggabungan Masker:**
  $$M_{\text{raw}} = M_{\text{YCrCb}} \land (M_{\text{HSV1}} \lor M_{\text{HSV2}})$$

### 3.3. Kalibrasi Warna Kulit Interaktif
Saat masuk ke mode kalibrasi (`[C]`), pemain mengarahkan telapak tangan ke dalam kotak target berukuran $220 \times 220\text{ px}$. Sistem menghitung rata-rata ($\mu$) dan deviasi standar ($\sigma$) dari saluran $Cr, Cb$ serta $H, S$:
$$Cr \in [\mu_{Cr} - 2.8\sigma_{Cr}, \, \mu_{Cr} + 2.8\sigma_{Cr}], \quad Cb \in [\mu_{Cb} - 2.8\sigma_{Cb}, \, \mu_{Cb} + 2.8\sigma_{Cb}]$$
$$H \in [\mu_H - 2.6\sigma_H, \, \mu_H + 2.6\sigma_H], \quad S \in [\mu_S - 2.6\sigma_S, \, \mu_S + 2.6\sigma_S]$$
Hasil kalibrasi disimpan otomatis di `save_data.json` dan dimuat ulang setiap sesi permainan.

### 3.4. Operasi Morfologi Citra
Membersihkan noise menggunakan elemen penstruktur elips $5 \times 5$ ($B$):
$$M_{\text{clean}} = \left( (M_{\text{isolated}} \circ B) \bullet B \right)$$
* **Opening ($M \circ B$):** Menghilangkan derau bintik putih di latar belakang.
* **Closing ($M \bullet B$):** Menutup lubang bayangan gelap di telapak tangan.

### 3.5. Ekstraksi Titik Berat (*Spatial Moments*) & EMA Smoothing
Titik tengah $(cX, cY)$ dihitung menggunakan momen spasial orde pertama:
$$cX_{\text{raw}} = \frac{m_{10}}{m_{00}}, \quad cY_{\text{raw}} = \frac{m_{01}}{m_{00}}$$
Diterapkan filter *Exponential Moving Average* (EMA) dengan faktor $\alpha = 0.65$:
$$\mathbf{C}_t = \alpha \mathbf{C}_{\text{raw}} + (1 - \alpha) \mathbf{C}_{t-1}$$
Menghasilkan bidikan stabil dan halus tanpa getaran mikro (*anti-jitter*).

---

## 4. Fitur Gameplay & Mekanika

### 4.1. Navigasi Menu & Sentuhan Virtual (*Virtual Touch*)
Selain tombol keyboard angka `[1-6]`, pemain dapat mengarahkan kursor tangan ke atas tombol UI. Kursor akan memunculkan indikator cincin progres melingkar selama $0.85\text{ detik}$, yang secara otomatis mengeklik tombol tersebut saat terisi penuh.

### 4.2. Pilihan Konfigurasi Grid
Pemain dapat menyesuaikan dimensi arena permainan:
* **Grid 2x2 (4 Lubang):** Arena ringkas, cocok untuk pemula.
* **Grid 2x3 (6 Lubang):** Tata letak standar seimbang (Default).
* **Grid 3x3 (9 Lubang):** Menantang koordinasi ruang vertikal dan horizontal.
* **Grid 3x4 (12 Lubang):** Arena luas ekstrem untuk refleks tingkat tinggi.

### 4.3. Mode Survival (Endless High Score)
* **Sistem Nyawa:** Pemain memiliki 3 HP (Hearts: ❤️❤️❤️). Jika mole kabur sebelum dipukul, nyawa berkurang 1.
* **Skala Intensitas Dinamis (Tier 1 s.d. 4):**
  - **Tier 1 (0–18s):** Kecepatan mole $1.60\text{s} - 1.95\text{s}$, 1 mole aktif, BGM santai (~110 BPM).
  - **Tier 2 (18–36s):** Kecepatan mole $1.25\text{s} - 1.55\text{s}$, 1 mole aktif, BGM aktif (~130 BPM).
  - **Tier 3 (36–54s):** Kecepatan mole $0.95\text{s} - 1.20\text{s}$, 2 mole simultan, BGM intens (~155 BPM).
  - **Tier 4 (>54s):** Kecepatan mole $0.60\text{s} - 0.85\text{s}$, 2 mole simultan, BGM ekstrem (~180 BPM).
* **Combo Multiplier:** Pukulan berturut-turut tanpa jeda mole kabur melipatgandakan perolehan poin ($x1, x2, x3, x4$).

### 4.4. Mode Level Progression (Level 1 - 10)
Sistem level bertingkat dengan target pukulan dan batas waktu spesifik:
* **Level 1 ("Pemanasan"):** Target 8 mole / 30 detik (Kecepatan: 1.70s - 2.00s)
* **Level 2 ("Refleks Cepat"):** Target 10 mole / 30 detik (Kecepatan: 1.50s - 1.80s)
* **Level 3 ("Lubang Lincah"):** Target 12 mole / 30 detik (Kecepatan: 1.35s - 1.60s)
* **Level 4 ("Fokus Ganda"):** Target 14 mole / 28 detik (Kecepatan: 1.20s - 1.45s)
* **Level 5 ("Irama Cepat"):** Target 16 mole / 28 detik (Kecepatan: 1.10s - 1.30s)
* **Level 6 ("Ujian Akurasi"):** Target 18 mole / 26 detik (Kecepatan: 1.00s - 1.20s)
* **Level 7 ("Serbuan Mole"):** Target 20 mole / 25 detik (Kecepatan: 0.90s - 1.10s)
* **Level 8 ("Badai Refleks"):** Target 22 mole / 25 detik (2 mole simultan, Kecepatan: 0.80s - 1.00s)
* **Level 9 ("Zona Ekstrem"):** Target 25 mole / 25 detik (2 mole simultan, Kecepatan: 0.70s - 0.90s)
* **Level 10 ("Grandmaster"):** Target 30 mole / 25 detik (2 mole simultan, Kecepatan: 0.55s - 0.75s)

> **Aturan Kunci:** Level $N+1$ hanya dapat dimainkan setelah Level $N$ berhasil diselesaikan (*cleared*).

---

## 5. Sintesis Audio Prosedural & BGM Adaptif

Seluruh audio dihasilkan dari formulasi matematis NumPy pada frekuensi sampling $22.050\text{ Hz}$:
* **Click / Navigasi:** Gelombang sinus $880\text{ Hz}$ dengan peluruhan eksponensial cepat ($\tau = 60$).
* **Whack / Bonk:** Sapuan frekuensi linier dari $440\text{ Hz}$ turun ke $90\text{ Hz}$ dengan harmonik kedua.
* **Miss (Kabur):** Gelombang kotak teredam $190\text{ Hz} \to 80\text{ Hz}$.
* **Level Clear Fanfare:** Arpeggio nada C mayor ($C_5, E_5, G_5, C_6$).
* **Game Over:** Arpeggio nada minor menurun ($440, 370, 311, 220\text{ Hz}$).
* **BGM Adaptif 4-Tingkat:** Ritme bassline dan ketukan drum terprogram yang otomatis beralih tempo dan intensitas mengikuti tensi permainan.

---

## 6. Petunjuk Menjalankan Program

### Persyaratan Pustaka
```bash
pip install -r requirements.txt
```
*(Memerlukan `opencv-python`, `numpy`, dan `pygame`)*

### Menjalankan Game
Dari folder utama repositori (`project-1`):
```bash
python whack-a-mole/main.py
```
Atau dari dalam sub-folder `whack-a-mole`:
```bash
cd whack-a-mole
python main.py
```

### Navigasi & Kontrol Keyboard
| Tombol | Fungsi |
| :--- | :--- |
| **`1` - `6`** | Memilih opsi menu pada Menu Utama (Survival, Level, Grid, Kalibrasi, Audio, Keluar) |
| **`1` - `9`, `0`** | Memilih Level 1 s.d. 10 pada layar Level Select |
| **`1` - `4`** | Memilih ukuran Grid pada layar Grid Select |
| **`C`** | Masuk/Keluar Mode Kalibrasi Warna Kulit |
| **`SPASI` / `ENTER`** | Mengunci kalibrasi warna kulit / Melanjutkan ke level berikutnya saat Level Clear |
| **`H` / `TAB`** | Mengganti orientasi zona bermain (Tangan Kanan, Tangan Kiri, Tangan Bawah) |
| **`R`** | Mengulang (*Restart*) sesi permainan saat Game Over |
| **`S`** | Menyalakan / mematikan master audio (*Toggle Sound*) |
| **`P`** | Menyalakan / mematikan jendela Picture-in-Picture (PiP) Masker Kulit |
| **`M`** / **`ESC`** | Kembali ke Menu Utama dari layar mana pun / Keluar |
