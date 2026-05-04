# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN/LSTM)

Implementasi Convolutional Neural Network (CNN) serta Recurrent Neural Network (RNN dan LSTM) dari nol menggunakan Python dan NumPy, untuk memenuhi spesifikasi Tugas Besar 2 IF3270 Pembelajaran Mesin 2025/2026.

Tugas ini mencakup:

* Implementasi forward propagation CNN, RNN, dan LSTM dari scratch
* Pelatihan model menggunakan Keras sebagai baseline
* Pengembangan pipeline *image captioning* berbasis encoder-decoder (CNN + LSTM/RNN)

---

## Struktur Repository

```
├── README.md
├── data/
│   ├── raw/
│   │   ├── intel_image_classification/
│   │   └── flickr8k/
│   ├── processed/
│   └── features/
│
├── notebooks/
│
├── doc/
│   └── laporan.pdf
│
└── src/
    ├── cnn/
    │   ├── layers/
    │   ├── models/
    │   └── utils/
    │   
    │
    └── rnn/
        ├── layers/
        ├── models/
        └── utils/
```

---

## Setup & Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dependensi

* Python 3.8+
* NumPy
* TensorFlow / Keras
* Matplotlib
* scikit-learn
* pandas
* tqdm
* nltk (untuk evaluasi captioning)
* jupyter

---

## Deskripsi Tugas

### 1. Convolutional Neural Network (CNN)

* Implementasi layer CNN dari scratch:

  * Conv2D (shared & non-shared / locally connected)
  * Pooling (max & average)
  * Flatten & Dense
  * Fungsi aktivasi (ReLU, Softmax)
* Pelatihan model CNN menggunakan Keras
* Eksperimen:

  * Variasi jumlah layer
  * Jumlah filter
  * Ukuran kernel
  * Jenis pooling
* Evaluasi menggunakan **macro F1-score**

---

### 2. Recurrent Neural Network & LSTM

* Implementasi dari scratch:

  * Embedding layer
  * Simple RNN cell
  * LSTM cell
  * Dense projection & output layer
* Preprocessing caption (tokenisasi, padding, vocabulary)
* Pelatihan model decoder menggunakan Keras
* Eksperimen:

  * Jumlah layer
  * Ukuran hidden state
* Evaluasi menggunakan:

  * BLEU-4 score
  * METEOR score

---

### 3. Image Captioning Pipeline

* Menggabungkan CNN encoder + RNN/LSTM decoder
* Feature extraction menggunakan pretrained CNN (InceptionV3 / VGG16)
* Implementasi inference dari raw image → caption
* Perbandingan:

  * RNN vs LSTM
  * Keras vs from scratch

---

## Cara Menjalankan Program

### Training CNN

```bash
python src/cnn/train_cnn.py
```

### Training RNN / LSTM

```bash
python src/rnn/train_rnn.py
python src/rnn/train_lstm.py
```

### Menjalankan Image Captioning

```bash
python src/pipeline/caption_generator.py
```

---

## Pembagian Tugas

| Nama     | NIM   | Tugas                                                              |
| -------- | ----- | ------------------------------------------------------------------ |
| <Nama 1> | <NIM> | <Job1>      |
| <Nama 2> | <NIM> | <Job2>              |
| <Nama 3> | <NIM> | <Job3> |

---

## Catatan

* Implementasi forward propagation CNN, RNN, dan LSTM dilakukan **dari scratch menggunakan NumPy**
* Keras digunakan hanya untuk pelatihan model dan pembanding
* Seluruh eksperimen dan evaluasi dicatat dalam laporan

---
