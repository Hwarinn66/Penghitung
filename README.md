# Dense Object Counter

Aplikasi **Dense Object Counter** untuk mendeteksi dan menghitung objek pada gambar/video menggunakan computer vision. Project ini menyediakan beberapa metode deteksi, termasuk OpenCV dan YOLO, serta mendukung inference berbasis tile untuk gambar beresolusi tinggi.

## Fitur

- Deteksi dan penghitungan objek secara otomatis.
- Dukungan detector berbasis **OpenCV** dan **YOLO**.
- Tiled inference untuk menangani gambar berukuran besar/dense object.
- Region of Interest (ROI) untuk membatasi area penghitungan.
- Antarmuka web berbasis Flask.
- Pemrosesan kamera/video dan snapshot.
- Penyimpanan hasil deteksi dan anotasi.
- Utility untuk persiapan dataset dan training YOLO.
- Konfigurasi melalui environment variable.

## Struktur Project

```text
dense-object-counter/
├── app.py
├── config.py
├── requirements.txt
├── requirements-base.txt
├── requirements-dev.txt
├── pytest.ini
│
├── camera/
│   └── camera_manager.py
│
├── counting/
│   ├── counter.py
│   └── roi.py
│
├── detection/
│   ├── base_detector.py
│   ├── factory.py
│   ├── opencv_detector.py
│   ├── tiled_inference.py
│   └── yolo_detector.py
│
├── training/
│   ├── dataset.yaml
│   ├── dataset_utils.py
│   ├── evaluate_counting.py
│   ├── prepare_tiles.py
│   └── train.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── app.js
│   └── style.css
│
├── dataset/
│   ├── images/
│   └── labels/
│
├── models/
├── output/
└── debug/
```

## Persyaratan

- Python 3.10+ direkomendasikan.
- pip
- Webcam/camera jika menggunakan mode kamera.
- Model YOLO jika menggunakan detector YOLO.

## Instalasi

Clone repository:

```bash
git clone https://github.com/Hwarinn66/Penghitung.git
cd Penghitung/dense-object-counter
```

Buat virtual environment:

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Untuk kebutuhan development/testing:

```bash
pip install -r requirements-dev.txt
```

## Menjalankan Aplikasi

Jalankan aplikasi Flask:

```bash
python app.py
```

Setelah server berjalan, buka alamat yang ditampilkan oleh Flask pada browser.

Jika aplikasi menggunakan konfigurasi melalui environment variable, sesuaikan nilai konfigurasi terlebih dahulu sesuai kebutuhan.

## Detector

Project mendukung beberapa pendekatan deteksi.

### OpenCV

Detector OpenCV dapat digunakan untuk skenario sederhana yang tidak membutuhkan model deep learning.

### YOLO

Detector YOLO digunakan untuk object detection berbasis model terlatih.

Letakkan model yang digunakan di folder:

```text
models/
```

Contoh:

```text
models/
└── best.pt
```

> File model `.pt` tidak disertakan dalam repository ini. Gunakan model hasil training sendiri atau model yang memang Anda miliki hak untuk menggunakannya.

## Tiled Inference

Untuk gambar dengan resolusi besar atau objek yang sangat padat, project menyediakan tiled inference.

Gambar akan dibagi menjadi beberapa tile, kemudian setiap tile diproses secara terpisah sebelum hasil deteksi digabungkan.

Pendekatan ini dapat membantu ketika objek terlalu kecil jika seluruh gambar diproses sekaligus.

Parameter tile dapat disesuaikan melalui konfigurasi pada project.

## Region of Interest (ROI)

ROI memungkinkan penghitungan hanya dilakukan pada area tertentu.

Dengan ROI, objek di luar area yang ditentukan dapat diabaikan sehingga hasil counting lebih relevan terhadap area yang ingin dianalisis.

## Dataset

Struktur dataset mengikuti format YOLO:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

File konfigurasi dataset tersedia di:

```text
training/dataset.yaml
```

Pastikan pasangan image dan label memiliki nama file yang sesuai.

Contoh:

```text
images/train/image001.jpg
labels/train/image001.txt
```

## Training Model

Utility training tersedia di folder `training/`.

Contoh menjalankan training:

```bash
python training/train.py
```

Sebelum training, periksa konfigurasi dataset dan parameter training pada script tersebut.

Utility tambahan:

- `prepare_tiles.py` — menyiapkan dataset berbasis tile.
- `dataset_utils.py` — utility pengolahan dataset.
- `evaluate_counting.py` — evaluasi hasil counting.

## Testing

Project menyediakan konfigurasi pytest melalui `pytest.ini`.

Jalankan test dengan:

```bash
pytest
```

## Output

Hasil pemrosesan dapat disimpan pada:

```text
output/
```

Folder `debug/` dapat digunakan untuk menyimpan hasil debugging atau visualisasi intermediate.

## Konfigurasi

Konfigurasi aplikasi dipusatkan pada:

```text
config.py
```

Sesuaikan konfigurasi detector, model, kamera, ROI, dan parameter inference sesuai kebutuhan deployment.

## Catatan Git

File besar seperti dataset, hasil training, dan model biasanya tidak sebaiknya disimpan langsung di repository Git.

Model seperti:

```text
best.pt
last.pt
```

serta dataset berukuran besar sebaiknya menggunakan Git LFS atau penyimpanan terpisah jika ukurannya terlalu besar untuk Git biasa.

## Alur Penggunaan

Secara umum alur aplikasi adalah:

```text
Image / Camera / Video
          │
          ▼
       Detector
          │
          ▼
   Object Detections
          │
          ▼
       ROI Filter
          │
          ▼
   Object Counting
          │
          ▼
 Annotation + Output
```

## Pengembangan

Struktur project dibuat modular agar detector, counting logic, ROI, camera management, dan training utilities dapat dikembangkan secara terpisah.

Jika ingin menambahkan detector baru, implementasikan interface dari:

```text
detection/base_detector.py
```

kemudian daftarkan detector tersebut melalui:

```text
detection/factory.py
```

## Lisensi

Tambahkan informasi lisensi project di bagian ini sesuai lisensi yang Anda gunakan.

## Author

**Hwarinn66**

Repository: https://github.com/Hwarinn66/Penghitung
