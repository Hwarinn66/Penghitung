# Web-Based Dense Object Counter

Aplikasi lokal Python/FastAPI untuk **snapshot counting** dari kamera HP IP Webcam, USB webcam, atau adapter kamera lain. Satu class saja: **`0 = object`**. Preview selalu hidup; inference hanya saat **HITUNG**. Source lengkap disertakan, tetapi **bobot model benda Anda dan dataset nyata belum disertakan**. Tidak ada angka hasil palsu atau model COCO yang diam-diam dianggap mengenali benda Anda.

## Mulai di sini

Gunakan Python **3.11 atau 3.12 64-bit**. Ekstrak arsip, masuk ke folder `dense-object-counter`, lalu:

```bash
python -m venv .venv
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi, gunakan Command Prompt atau jalankan `.venv\Scripts\python.exe` secara langsung. Linux/macOS:

```bash
source .venv/bin/activate
```

Install semua dependensi, termasuk YOLO:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Untuk fase kamera/web terlebih dahulu, instalasi lebih ringan tersedia:

```bash
pip install -r requirements-base.txt
```

Jalankan IP Webcam di HP, pilih kamera utama belakang **1×**, mulai server kamera. PC dan HP harus berada di Wi-Fi/LAN yang sama. Pastikan alamat video yang ditampilkan aplikasi HP bisa dibuka di PC. Contoh alamat saja, **ganti IP sesuai HP Anda**:

```bash
python app.py --cam "http://192.168.1.20:8080/video" --detector opencv
```

Buka **http://127.0.0.1:8000**. Mode awal ini menguji kamera, snapshot, ROI, serta counting warna. Default HSV memilih benda **biru**; jika benda Anda bukan biru, ubah HSV di `config.py`. Hasil baseline belum layak dianggap hitungan benda rapat.

Setelah model hasil fine-tuning tersedia pada `models/best.pt`:

```bash
python app.py --cam "http://192.168.1.20:8080/video"
```

Ganti ke USB webcam tanpa mengubah detector:

```bash
python app.py --cam 0
```

**Hanya satu proses server.** Gunakan perintah di atas, jangan `--reload` atau beberapa worker: setiap proses akan memiliki kamera dan model sendiri. Default bind `127.0.0.1` untuk PC lokal. Jika sengaja diakses perangkat lain pada jaringan privat, gunakan `--host 0.0.0.0` dan buka IP PC:8000. MVP tidak menyediakan login; jangan mempublikasikan port ini ke internet.

## Yang sudah diimplementasikan

- Satu `CameraManager`: satu worker pemilik `VideoCapture`, frame copy dengan lock, preview JPEG bersama untuk semua browser, deteksi frame basi, retry 2 detik, release saat shutdown normal.
- Dashboard responsif HTML/CSS/JavaScript, status polling 2 detik, live MJPEG, HITUNG/RESET, status model, waktu, resolusi aktual, hasil bernomor dan tautan resolusi penuh.
- Snapshot stream resolusi asli atau endpoint still opsional; tidak mengarang endpoint snapshot HP.
- ROI configurable dengan crop **sebelum detector**; area di luar crop tidak diberikan ke model. Preview menandai ROI.
- Adapter `BaseDetector`, OpenCV baseline, YOLO satu class, tile native 1024 px dengan overlap 15%, transform koordinat, NMS global, pemeriksaan fragment pada seam.
- `max_det=2000` per tile, tanpa batas global 300/500. Ini batas komputasi, bukan jaminan akurasi pada 2000 benda.
- Snapshot penuh dan hasil tersimpan dengan timestamp UTC mikrodetik + UUID. JSON menyimpan box, confidence, timing, warning, dan metadata ROI.
- Debug snapshot, crop ROI, semua tile, raw/merged detections, dan final image. Default debug mati.
- Dataset YAML, validator label, preparation tile train/val, training pretrained, evaluasi count per gambar dan per kepadatan.
- Tes regresi untuk lifecycle kamera, API, concurrency, ROI, tile, dataset, dan lebih dari 500 kandidat.

## Struktur proyek

```text
dense-object-counter/
├── app.py
├── config.py
├── requirements.txt
├── requirements-base.txt
├── requirements-dev.txt
├── README.md
├── camera/
│   ├── __init__.py
│   └── camera_manager.py
├── counting/
│   ├── __init__.py
│   ├── counter.py
│   └── roi.py
├── detection/
│   ├── __init__.py
│   ├── base_detector.py
│   ├── factory.py
│   ├── opencv_detector.py
│   ├── yolo_detector.py
│   └── tiled_inference.py
├── training/
│   ├── __init__.py
│   ├── dataset.yaml
│   ├── dataset_utils.py
│   ├── prepare_tiles.py
│   ├── train.py
│   └── evaluate_counting.py
├── templates/index.html
├── static/
│   ├── style.css
│   └── app.js
├── dataset/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── docs/
│   ├── CAMERA_AND_CALIBRATION.md
│   ├── DATASET_AND_TRAINING.md
│   ├── ARCHITECTURE_AND_API.md
│   └── VALIDATION.md
├── tests/
├── models/                    # letakkan best.pt hasil training
├── output/                    # dibuat/diisi saat counting
└── debug/                     # per-run jika DEBUG=True
```

`runs/` dan `dataset_tiles/` dibuat hanya jika training/evaluasi/preparation dijalankan. Semua path default diturunkan dari folder proyek, bukan current directory terminal.

## Tahapan implementasi dan acceptance check

Implementasi semua tahap tersedia secara modular, tetapi **aktifkan dan verifikasi berurutan**. Jangan memakai model sebelum kondisi kamera dan ROI benar.

| Fase | Jalankan / lakukan | Dinyatakan siap jika |
|---|---|---|
| 1 — Camera | `python app.py --cam URL --detector opencv` | Resolusi aktual benar, feed hidup. Matikan IP Webcam lalu hidupkan; server tidak mati dan kamera reconnect. |
| 2 — Web | Buka dashboard; `/api/status` | Status berubah sesuai kamera, preview stabil, HITUNG mati jika terputus. |
| 3 — Snapshot | Tekan HITUNG | `output/original_*.jpg` sesuai kondisi saat itu, tetap resolusi penuh. |
| 4 — ROI | Tandai 90×90 cm pada alas 100×100; edit config | Kotak preview cocok tanda fisik, benda luar ROI tidak dihitung. |
| 5 — Baseline | Kalibrasi HSV dan area contour | Benda terpisah terdeteksi; dokumentasikan kegagalan ketika bersentuhan. |
| 6 — Dataset | Kumpulkan, anotasi, split berdasarkan sesi | Label satu class; negative memiliki `.txt` kosong; setiap instance terpisah diberi box. |
| 7 — Training | `python training/train.py` | Training selesai, best weights tersalin ke `models/best.pt`. |
| 8 — YOLO | Jalankan tanpa `--detector opencv` | Detector loaded, orientasi berbeda tetap class object, hasil visual diperiksa. |
| 9 — Tiling | Default `TILED_INFERENCE=True`; uji posisi seam | Jumlah tetap benar saat kelompok benda bergeser melintasi seam. |
| 10 — Evaluation | Evaluasi full-resolution test set | Laporan count error tersedia untuk 10/50/100/200/300+ dan negative. |

Jika model belum ada, dashboard tetap hidup: status `detector_unavailable`, preview dapat dipakai, HITUNG dinonaktifkan. Setelah menyalin model, restart backend untuk memuatnya.

## Workflow operator

1. Hidupkan IP Webcam; jalankan backend; buka dashboard.
2. Pastikan Camera **CONNECTED**, detector siap, ROI terkalibrasi.
3. Sebarkan benda menjadi satu lapisan di dalam tanda counting. Boleh bersentuhan dan overlap ringan.
4. Pindahkan benda yang melintasi garis ROI; tunggu benda diam dan tangan keluar dari area.
5. Tekan **HITUNG** sekali. Tombol menjadi **MEMPROSES...**. Jangan memindahkan benda selama pengambilan snapshot.
6. Lihat jumlah integer dalam PCS dan gambar bernomor; buka resolusi penuh untuk memeriksa miss/duplicate.
7. **RESET** menghapus hasil aktif, tidak mematikan kamera dan tidak menghapus arsip gambar.

Jangan mengasumsikan waktu 2.31 detik: waktu aktual bergantung CPU/GPU, resolusi, jumlah tile, model, dan penyimpanan. First inference bisa lebih lambat karena inisialisasi backend.

## Dokumen operasional

- [Kamera, ROI, kalibrasi parameter, troubleshooting](docs/CAMERA_AND_CALIBRATION.md)
- [Dataset, annotation, training, evaluasi count](docs/DATASET_AND_TRAINING.md)
- [Arsitektur, API, koordinat, batas sistem](docs/ARCHITECTURE_AND_API.md)
- [Hasil dan batas pengujian implementasi](docs/VALIDATION.md)

## Batas fisik dan akurasi

Satu contour bukan satu benda jika benda berdempetan. Karena itu OpenCV hanya baseline. YOLO dilatih agar **setiap benda fisik menghasilkan satu instance**, dengan semua orientasi sebagai class object.

Benda yang **tertutup sepenuhnya** oleh benda lain tidak dapat diketahui jumlahnya secara pasti dari satu kamera atas. Gunakan satu lapisan; hindari tumpukan bertingkat. NMS/tiling tidak memulihkan informasi yang tidak terlihat. Angka yang dihasilkan adalah **jumlah detection**, sehingga akurasi harus dibuktikan dengan test set nyata dan hitungan manual.

Jika objek terlalu kecil dalam piksel, menaikkan confidence atau mengganti NMS tidak mengembalikan detail. Perbaiki resolusi/fokus/pencahayaan, kurangi area efektif, atau gunakan setup optik yang sesuai. Tiling mempertahankan detail yang memang tersedia, bukan membuat detail baru.

## Tes pengembang

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tes otomatis menggunakan kamera/detector test double atau gambar sintetis; tidak memerlukan HP, `best.pt`, atau GPU. Tes integrasi HTTP memakai server MJPEG lokal. Lihat `docs/VALIDATION.md` untuk hasil yang benar-benar dijalankan. Sesudah memilih kombinasi hardware/PyTorch yang cocok, rekam environment dengan `pip freeze > requirements-local-lock.txt`.

## Referensi

- Mekanisme kamera yang diminta: https://github.com/Hwarinn66/Smart-Trash . Repository tidak berhasil diakses saat implementasi, sehingga tidak ada klaim penyalinan kode; pola URL → `VideoCapture` → read/release/retry mengikuti spesifikasi Anda.
- Ultralytics prediction dan parameter `imgsz`, `iou`, `max_det`: https://docs.ultralytics.com/modes/predict/
- Pretrained YOLO11: https://docs.ultralytics.com/models/yolo11/
- Training: https://docs.ultralytics.com/modes/train/
- Dataset detection: https://docs.ultralytics.com/datasets/detect/
- OpenCV camera properties/backend timeouts: https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html

Dependency YOLO dipin ke `ultralytics==8.3.161` dengan YOLO11; proyek ini tidak bergantung pada perubahan API model terbaru. Periksa ketentuan lisensi Ultralytics sebelum mendistribusikan aplikasi atau model dalam produk Anda.
