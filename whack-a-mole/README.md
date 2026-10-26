# Laporan Teknis Proyek 1 - Tipe 3: Whack-A-Mole Virtual Mini-Game (Pure OpenCV)

Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)  
Topik: *Virtual Interactive Whack-A-Mole Game with Adaptive Audio, Multi-Grid, Survival & Level Progression*  
Berkas Utama: [`main.py`](main.py) | [`game_engine.py`](game_engine.py) | [`hand_tracker.py`](hand_tracker.py) | [`audio_system.py`](audio_system.py) | [`ui_renderer.py`](ui_renderer.py) | [`entities.py`](entities.py) | [`save_manager.py`](save_manager.py) | [`config.py`](config.py)

---

## 1. Pendahuluan

**Whack-A-Mole Virtual** adalah game arcade interaktif berbasis *Computer Vision* murni (**OpenCV `cv2` & NumPy**) tanpa pustaka *deep learning* (tanpa MediaPipe, YOLO, TensorFlow, atau PyTorch). Pemain mengontrol seluruh navigasi menu dan aksi pemukulan mole secara *hands-free* di depan webcam melalui pelacakan titik berat (*centroid*) telapak tangan.

Proyek ini dilengkapi dengan:
1. **Main Menu Interaktif**: Navigasi menggunakan keyboard atau **Virtual Touch** (arahkan kursor tangan di atas tombol selama ~0.85 detik).
2. **Kustomisasi Grid**: Pilihan ukuran grid lubang $2 \times 2$ (4 lubang), $2 \times 3$ (6 lubang), $3 \times 3$ (9 lubang), dan $3 \times 4$ (12 lubang).
3. **Mode Permainan Ganda**:
   - **Mode Survival (Endless High Score)**: Intensitas meningkat secara dinamis seiring durasi bertahan (kecepatan mole menyusut hingga $0.55\text{s}$, kemunculan mole ganda, sistem nyawa 3 HP, dan *multiplier combo*).
   - **Mode Level Progression (Level 1 - 10)**: Sistem perkembangan 10 level bertahap. Untuk membuka level selanjutnya, pemain **wajib menuntaskan level sebelumnya**. Progres tersimpan permanen di `save_data.json`.
4. **Sintesis Audio Prosedural & BGM Dinamis Adaptif**: Seluruh efek suara (klik tombol, pukulan *bonk*, mole kabur, *level clear fanfare*, *game over*) serta 4 tingkat loop musik latar *chiptune* disintesis secara matematis menggunakan rumus gelombang NumPy tanpa file audio eksternal. Musik in-game beradaptasi semakin intens dan cepat saat intensitas permainan meningkat.

---

## 2. Arsitektur Modular & Desain Sistem

Kode dibangun dengan arsitektur bersih (*Clean Code & Separation of Concerns*):

```text
whack-a-mole/
│
├── config.py           # Parameter konstan: batas warna HSV, ukuran frame, definisi 10 level, warna UI
├── audio_system.py     # Generator gelombang audio sintetis (NumPy stereo PCM) & BGM loop adaptif
├── save_manager.py     # Persistensi data progres: level unlock, rekor survival, dan preferensi grid
├── hand_tracker.py     # Modul Vision: segmentasi HSV dual-range, morfologi, kontur & centroid EMA
├── entities.py         # Model objek: Hole, Mole (multi-mole), HitEffect, ScoreManager, SurvivalTracker, Button
├── ui_renderer.py      # Modul visualisasi OpenCV: HUD Survival/Level, Mole kartun, modal dialog, & PiP
├── game_engine.py      # Mesin utama: State Machine (MENU, LEVEL_SELECT, GRID_SELECT, PLAYING, GAMEOVER)
├── main.py             # Entry point aplikasi: loop kamera, event hotkey, dan eksekusi
├── save_data.json      # File penyimpanan lokal progres pemain (dibuat otomatis)
└── README.md           # Laporan teknis dan dokumentasi lengkap
```

---

## 3. Metodologi Pengolahan Citra Digital (PCD)

### 3.1. Segmentasi Warna Kulit HSV Dual-Range
Memisahkan krominans dari pencahayaan menggunakan model HSV:
* **Rentang 1 (Oranye–Kuning & Merah Muda):**
  $$H \in [0, 22], \quad S \in [35, 190], \quad V \in [60, 255]$$
* **Rentang 2 (Pembungkus Rona Merah Pekat / Wrap-Around):**
  $$H \in [170, 180], \quad S \in [35, 190], \quad V \in [60, 255]$$
* **Penggabungan Masker Biner:**
  $$M_{\text{raw}} = M_1 \lor M_2$$

### 3.2. Operasi Morfologi Citra
Membersihkan noise menggunakan elemen penstruktur elips $5 \times 5$ ($B$):
$$M_{\text{clean}} = \left( (M_{\text{raw}} \circ B) \bullet B \right) \oplus B$$
* **Opening ($M \circ B$):** Membuang derau garam (*salt noise*) pada latar belakang.
* **Closing ($M \bullet B$):** Menutup celah gelap (*pepper noise*) di permukaan telapak tangan.
* **Dilasi ($M \oplus B$):** Mempertegas batas tepi kontur.

### 3.3. Ekstraksi Titik Berat (*Spatial Moments*) & EMA Smoothing
Titik tengah $(cX, cY)$ dihitung menggunakan momen spasial orde pertama:
$$cX_{\text{raw}} = \frac{m_{10}}{m_{00}}, \quad cY_{\text{raw}} = \frac{m_{01}}{m_{00}}$$
Diterapkan filter *Exponential Moving Average* (EMA) dengan faktor $\alpha = 0.65$:
$$\mathbf{C}_t = \alpha \mathbf{C}_{\text{raw}} + (1 - \alpha) \mathbf{C}_{t-1}$$
Teknik ini menghasilkan kursor bidik yang sangat mulus tanpa getaran mikro (*anti-jitter*).

### 3.4. Heuristik Multi-Candidate Scoring Anti-Wajah & Temporal Lock
Untuk mencegah kursor terkunci pada kepala atau wajah pemain di tengah layar:
1. **Penyaringan Multi-Kandidat**: Menilai seluruh kontur yang memenuhi ambang batas $1.800 \le \text{Area} \le 75.000\text{ px}$.
2. **Penalti Zona Wajah**: Kontur dengan pusat massa di area atas-tengah ($y < 0.38H$ dan $|x - 0.5W| < 0.20W$) yang berbentuk oval pekat ($\text{Solidity} > 0.82$) diberikan penalti besar ($-4.500\text{ poin}$).
3. **Bonus Zona Bermain & Geometri Tangan**: Kontur di zona bermain aktif serta memiliki cacat kecembungan (*convexity defects* celah jari) mendapat bonus $+2.000$ hingga $+3.500\text{ poin}$.
4. **Temporal Hand Tracking Lock**: Kontur yang dekat dengan posisi tangan frame sebelumnya ($\Delta d < 200\text{ px}$) diberikan bonus pelacakan tinggi sehingga kursor mengunci gerakan tangan saat di-*drag* ke sudut layar mana pun tanpa melompat ke wajah.
5. **Mode Kalibrasi Cepat (Tombol `C`)**: Pemain dapat menekan tombol `C` untuk mengambil sampel warna kulit telapak tangan pada kondisi pencahayaan ruangan saat bermain.

---

## 4. Fitur Gameplay & Mekanika

### 4.1. Navigasi Menu & Sentuhan Virtual (*Virtual Touch*)
Selain menekan tombol keyboard, pemain dapat mengarahkan kursor tangan ke atas tombol UI. Kursor akan memunculkan lingkaran meteran pengisian progres melingkar selama $0.85\text{ detik}$, yang secara otomatis mengeklik tombol tersebut saat terisi penuh.

### 4.2. Pilihan Konfigurasi Grid
Pemain dapat menyesuaikan dimensi arena permainan:
* **Grid 2x2 (4 Lubang):** Arena ringkas, cocok untuk pemula.
* **Grid 2x3 (6 Lubang):** Tata letak klasik seimbang (Default).
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
* **Level 1 ("Pemanasan"):** Target 8 mole / 30 detik (Kecepatan mole: 1.70s - 2.00s)
* **Level 2 ("Refleks Cepat"):** Target 10 mole / 30 detik (Kecepatan mole: 1.50s - 1.80s)
* **Level 3 ("Lubang Lincah"):** Target 12 mole / 30 detik (Kecepatan mole: 1.35s - 1.60s)
* **Level 4 ("Fokus Ganda"):** Target 14 mole / 28 detik (Kecepatan mole: 1.20s - 1.45s)
* **Level 5 ("Irama Cepat"):** Target 16 mole / 28 detik (Kecepatan mole: 1.10s - 1.30s)
* **Level 6 ("Ujian Akurasi"):** Target 18 mole / 26 detik (Kecepatan mole: 1.00s - 1.20s)
* **Level 7 ("Serbuan Mole"):** Target 20 mole / 25 detik (Kecepatan mole: 0.90s - 1.10s)
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
| **`1` - `5`** | Memilih opsi menu pada Menu Utama |
| **`1` - `9`, `0`** | Memilih Level 1 s.d. 10 pada layar Level Select |
| **`1` - `4`** | Memilih ukuran Grid pada layar Grid Select |
| **`C`** | Masuk ke Mode Kalibrasi Warna Kulit (Arahkan tangan ke kotak hijau lalu tekan SPASI) |
| **`SPASI`** | Melanjutkan ke level berikutnya saat Level Clear / Kunci kalibrasi |
| **`R`** | Mengulang (*Restart*) sesi permainan saat Game Over |
| **`S`** | Menyalakan / mematikan master audio (*Toggle Sound*) |
| **`P`** | Menyalakan / mematikan jendela Picture-in-Picture (PiP) Masker Kulit |
| **`M`** / **`ESC`** | Kembali ke Menu Utama dari layar mana pun / Keluar |
