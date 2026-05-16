# Tugas Besar 2 IF3270 Pembelajaran Mesin

Implementasi CNN serta RNN/LSTM untuk Tugas Besar 2 IF3270 Pembelajaran Mesin. Implementasi `from scratch` difokuskan pada forward propagation menggunakan Python dan NumPy, sedangkan Keras digunakan untuk training baseline dan sumber bobot pembanding.

## Ruang Lingkup

- CNN untuk klasifikasi citra Intel Image Classification.
- RNN/LSTM untuk image captioning Flickr8k.
- Perbandingan hasil Keras dan implementasi `from scratch`.
- Evaluasi CNN menggunakan Macro F1-score.
- Evaluasi captioning menggunakan BLEU-4 dan METEOR.
- Batch inference untuk decoder captioning.

## Struktur Repository

```text
.
├── data/
│   ├── features/
│   ├── models/
│   └── raw/
├── docs/
│   ├── main.tex
│   ├── main.pdf
│   ├── bab1.tex
│   ├── bab2.tex
│   ├── bab3.tex
│   └── bab4.tex
├── notebooks/
│   ├── cnn.ipynb
│   └── rnn/
│       ├── rnn_lstm.ipynb
│       └── rnn-lstm_new.ipynb
├── src/
│   ├── cnn/
│   │   ├── layers/
│   │   └── utils/
│   └── rnn/
│       ├── layers/
│       ├── utils/
│       ├── train_rnn.py
│       ├── train_lstm.py
│       └── train_utils.py
├── requirements.txt
└── README.md
```

Catatan: untuk RNN/LSTM, gunakan `notebooks/rnn/rnn-lstm_new.ipynb` sebagai notebook hasil terbaru. Notebook `rnn_lstm.ipynb` berisi hasil evaluasi lama sebelum fix.

## Setup

```bash
python -m venv venv
```

Aktivasi environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notebook Eksperimen

### CNN

Notebook utama:

```text
notebooks/cnn.ipynb
```

Isi utama:

- Implementasi forward propagation layer CNN dari scratch.
- Training CNN dengan Keras untuk 16 konfigurasi.
- Variasi jumlah layer, jumlah filter, ukuran kernel, dan jenis pooling.
- Evaluasi Macro F1-score.
- Perbandingan Keras vs `from scratch`.
- Perbandingan `Conv2D` shared weights vs `LocallyConnected2D` non-shared weights.

### RNN/LSTM

Notebook utama terbaru:

```text
notebooks/rnn/rnn-lstm_new.ipynb
```

Isi utama:

- Feature extraction gambar menggunakan InceptionV3 pretrained.
- Caption preprocessing: tokenisasi, vocabulary, dan padding.
- Training 6 variasi RNN dan 6 variasi LSTM.
- Konversi bobot Keras ke layer NumPy `from scratch`.
- Evaluasi BLEU-4 dan METEOR.
- Perbandingan RNN vs LSTM.
- Perbandingan Keras vs `from scratch`.
- Analisis variasi `max_length`.
- Analisis kualitatif contoh caption.

## Implementasi From Scratch

### CNN

Layer tersedia di `src/cnn/layers/`:

- `Conv2D`
- `LocallyConnected2D`
- `MaxPooling2D`
- `AveragePooling2D`
- `GlobalMaxPooling2D`
- `GlobalAveragePooling2D`
- `Flatten`
- `Dense`
- Aktivasi ReLU dan Softmax

Utility tersedia di `src/cnn/utils/`:

- image loader
- batch loader
- feature extractor

### RNN/LSTM

Layer tersedia di `src/rnn/layers/`:

- `Embedding`
- `SimpleRNNCell` dan `SimpleRNN`
- `LSTMCell` dan `LSTM`
- `StackedRNN`
- `Dense`
- Aktivasi sigmoid, tanh, dan softmax

Utility tersedia di `src/rnn/utils/`:

- caption preprocessing
- feature extraction InceptionV3
- greedy decoder
- batch inference melalui `GreedyCaptionDecoder.generate_batch`
- scoring BLEU-4 dan METEOR
- helper konversi bobot Keras

## Training Script RNN/LSTM

Training decoder juga dapat dijalankan lewat script berikut dengan argumen dataset dan output yang sesuai:

```bash
python src/rnn/train_rnn.py --captions-path <path/to/captions.txt> --image-dir <path/to/images> --output-dir data/models/rnn
python src/rnn/train_lstm.py --captions-path <path/to/captions.txt> --image-dir <path/to/images> --output-dir data/models/rnn
```

Untuk eksperimen lengkap yang sudah menghasilkan tabel dan plot evaluasi, gunakan notebook RNN/LSTM terbaru.

## Laporan

Laporan LaTeX berada di folder `docs/`.

File utama:

```text
docs/main.tex
```

PDF hasil kompilasi:

```text
docs/main.pdf
```

Kompilasi manual dari folder `docs/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Status Bonus

Sudah ada:

- Batch inference decoder RNN/LSTM melalui `generate_batch`.

Belum ada / belum menjadi bagian final:

- Visualisasi intermediate feature maps CNN.
- Grad-CAM CNN.
- Beam search decoder.
- Script end-to-end raw image ke caption di `src/pipeline/`.

## Pembagian Tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13523092 | Muhammad Izzat Jundy | Notebook RNN/LSTM dan laporan |
| 13523094 | Zulfaqqar Nayaka Athadiansyah | Notebook CNN dan laporan |
| 13523098 | Muhammad Adha Ridwan | Implementasi model from scratch, feature extractor, batch inference, dan laporan |

