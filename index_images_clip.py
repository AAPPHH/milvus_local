import os
import torch
import clip  # Stelle sicher, dass das clip-Package installiert ist (pip install git+https://github.com/openai/CLIP.git)
from PIL import Image
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import json

class CLIPImageIndexer:
    def __init__(self, 
                 image_dir="./images", 
                 collection_name="image_vectors",
                 dim=768,
                 batch_size=500,
                 mapping_path="id_to_path.json",
                 host="localhost",
                 port="19530"):
        
        self.image_dir = image_dir
        self.collection_name = collection_name
        self.dim = dim
        self.batch_size = batch_size
        self.mapping_path = mapping_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delete_existing = True  # Setze auf False, um die Collection nicht zu löschen

        self.id_topath = {}

        self._load_mapping()
        self._setup_model()
        self._connect_milvus(host, port)
        self._prepare_collection()

    def _load_mapping(self):
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r") as f:
                self.id_to_path = json.load(f)
        else:
            self.id_to_path = {}


    def _setup_model(self):
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def _connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)

    def _prepare_collection(self):
        if self.collection_name in utility.list_collections() and self.delete_existing:
            Collection(self.collection_name).drop()
            print(f"🗑️  Bestehende Collection '{self.collection_name}' gelöscht.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, description="CLIP Image Embeddings mit Label")
        self.collection = Collection(name=self.collection_name, schema=schema)

        self.collection.create_index("embedding", {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        })
        self.collection.load()

    def _image_to_vector(self, image_path):
        img = Image.open(image_path).convert("RGB")
        # Verwende CLIPs Preprocessing, um das Bild in den erforderlichen Tensor zu konvertieren
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        features = features.cpu().numpy().flatten()
        # Normiere den Vektor
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def index_images(self):
        print("🔍 Durchsuche Bildverzeichnis...")
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, file))

        ids, vectors, labels = [], [], []
        id_to_path = {}

        print("🧠 Vektorisierung startet...")

        for idx, path in enumerate(image_paths):
            try:
                vec = self._image_to_vector(path)
                # Extrahiere das Label aus dem Verzeichnisnamen (z.B. "n02085620-Chihuahua")
                folder = os.path.basename(os.path.dirname(path))
                label = folder.split("-")[1] if "-" in folder else folder
                ids.append(idx)
                vectors.append(vec)
                labels.append(label)
                id_to_path[idx] = path
                print(f"✅ [{idx}] {os.path.basename(path)} → Label: {label}")
            except Exception as e:
                print(f"⚠️ Fehler bei {path}: {e}")

        print(f"\n🚚 Sende Vektoren in Batches von {self.batch_size}...")

        for i in range(0, len(ids), self.batch_size):
            batch_ids = ids[i:i + self.batch_size]
            batch_vecs = vectors[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]
            self.collection.insert([batch_ids, batch_vecs, batch_labels])
            print(f"✅ Batch {i//self.batch_size + 1}: {len(batch_ids)} Einträge eingefügt")

        self.collection.flush()
        print(f"\n🚀 Insgesamt {len(ids)} Vektoren erfolgreich eingefügt.")

        with open(self.mapping_path, "w") as f:
            import json
            json.dump(id_to_path, f)
        print(f"💾 ID-Mapping gespeichert unter: {self.mapping_path}")


if __name__ == "__main__":
    indexer = CLIPImageIndexer()
    indexer.index_images()
    print("✅ Alle Bilder erfolgreich indiziert und gespeichert.")
