
# 🧠 CLIP Image Search with Milvus

Ein einfaches System zur Vektorisierung und Suche von Bildern basierend auf OpenAI's [CLIP](https://github.com/openai/CLIP) und [Milvus](https://milvus.io/).

## 📦 Features

- Vektorisierung von Bildern mit CLIP (`ViT-L/14`)
- Speicherung in einer Milvus-Vektordatenbank
- Suche per Bild oder Text
- Visualisierung der Top-K Ergebnisse
- Docker-Compose Setup für Milvus + Abhängigkeiten

## 🚀 Setup

### 1. Milvus starten

```bash
docker-compose up -d
```

Dies startet:
- **Milvus Standalone**
- **MinIO** (Objektspeicher)
- **etcd** (Konfigurationsspeicher)

### 2. Python Umgebung vorbereiten

Am besten via `mamba` oder `conda`:

```bash
mamba create -n clipsearch python=3.10
mamba activate clipsearch
pip install -r requirements.txt
```

Falls `requirements.txt` fehlt, installiere manuell:

```bash
pip install torch torchvision matplotlib pillow numpy pymilvus git+https://github.com/openai/CLIP.git
```

## 🏷️ Bilder indizieren

Lege deine Bilder in `./images` ab. Die Ordnernamen sollten Labels enthalten, z. B.:

```
images/
└── n02085620-Chihuahua/
    ├── img1.jpg
    └── img2.jpg
```

Dann führe aus:

```bash
python index_images_clip.py
```

Dies:
- vektorisiert alle `.jpg`/`.png`-Bilder
- speichert Embeddings in Milvus
- erstellt eine `id_to_path.json` Mapping-Datei

## 🔍 Suche durchführen

### Textbasierte Suche:

```bash
python search_clip.py
```

Im Code kannst du `QUERY = "dein text"` anpassen.

### Bildbasierte Suche:

```bash
python search_clip.py /path/to/image.jpg
```

## ⚙️ Konfiguration

In `index_images_clip.py` und `search_clip.py` kannst du Host, Port, Collection-Name und weitere Parameter anpassen.

## 🧹 Milvus stoppen

```bash
docker-compose down
```

## 📁 Projektstruktur

```
.
├── docker-compose.yml          # Milvus + MinIO + etcd Setup
├── images/                     # Eingabebilder (nach Label sortiert)
├── index_images_clip.py        # Indexierung von Bildern
├── search_clip.py              # Suche mit Bild oder Text
├── id_to_path.json             # Mapping Datei
└── README.md                   # Diese Datei
```
