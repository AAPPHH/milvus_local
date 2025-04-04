import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

class ImageIndexer:
    def __init__(self, 
                 image_dir="./images", 
                 collection_name="image_vectors",
                 dim=2048,
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

        self._setup_model()
        self._connect_milvus(host, port)
        self._prepare_collection()

    def _setup_model(self):
        # Verwende moderne Weight-Auswahl + automatisch passende Transforms
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval().to(self.device)
        self.model = model
        self.transform = weights.transforms()

    def _connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)

    def _prepare_collection(self):
        if self.collection_name in utility.list_collections():
            Collection(self.collection_name).drop()
            print(f"🗑️  Bestehende Collection '{self.collection_name}' gelöscht.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, description="Image Embeddings mit Label")
        self.collection = Collection(name=self.collection_name, schema=schema)

        self.collection.create_index("embedding", {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        })
        self.collection.load()

    def _image_to_vector(self, image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor).squeeze().cpu().numpy()
        return features / np.linalg.norm(features)

    def index_images(self):
        print("🔍 Durchsuche Bildverzeichnis...")
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))

        ids, vectors, labels = [], [], []
        id_to_path = {}

        print("🧠 Vektorisierung startet...")

        for idx, path in enumerate(image_paths):
            try:
                vec = self._image_to_vector(path)
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
            print(f"✅ Batch {i//self.batch_size + 1}: {len(batch_ids)} eingefügt")

        self.collection.flush()
        print(f"\n🚀 Insgesamt {len(ids)} Vektoren erfolgreich eingefügt.")

        with open(self.mapping_path, "w") as f:
            json.dump(id_to_path, f)
        print(f"💾 ID-Mapping gespeichert unter: {self.mapping_path}")


if __name__ == "__main__":
    indexer = ImageIndexer()
    indexer.index_images()
    print("✅ Alle Bilder erfolgreich indiziert und gespeichert.")
