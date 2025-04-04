import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pymilvus import connections, Collection
from torchvision import models
from torchvision.models import ResNet50_Weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ImageSearcher:
    def __init__(self, 
                 collection_name="image_vectors", 
                 mapping_path="id_to_path.json",
                 host="localhost",
                 port="19530"):
        self.collection_name = collection_name
        self.mapping_path = mapping_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modell und Transformation ähnlich wie im Indexer
        self._setup_model()
        self._connect_milvus(host, port)
        
        # Bestehende Collection laden (diese wurde vorher vom Indexer angelegt)
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        # Mapping von IDs zu Bildpfaden laden (wird beim Indizieren gespeichert)
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r") as f:
                self.id_to_path = json.load(f)
        else:
            self.id_to_path = {}
            print(f"⚠️  Mapping-Datei '{self.mapping_path}' nicht gefunden.")

    def _setup_model(self):
        # Verwende dasselbe Modell und dieselben Gewichte wie im Indexer
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval().to(self.device)
        self.model = model
        self.transform = weights.transforms()

    def _connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)

    def image_to_vector(self, image_path):
        """Wandelt ein Bild in einen normalisierten Vektor um."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def search(self, query_image, top_k=5, visualize=True):
        """
        Führt eine Suche in der Milvus-Collection durch, basierend auf dem Query-Bild.
        Optional wird das Ergebnis visuell mit matplotlib dargestellt.
        """
        query_vec = self.image_to_vector(query_image).tolist()

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "L2", "nprobe": 10},
            limit=top_k,
            output_fields=["label"]
        )

        print(f"🔍 Top {top_k} ähnliche Bilder zu '{query_image}':\n")

        if visualize:
            fig, axes = plt.subplots(1, top_k, figsize=(15, 5))

            for i, hit in enumerate(results[0]):
                label = hit.entity.get("label")
                distance = hit.distance
                # Da JSON-Schlüssel als Strings gespeichert werden,
                # wird hit.id als String abgefragt.
                image_path = self.id_to_path.get(str(hit.id), self.id_to_path.get(hit.id))
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    axes[i].set_title(f"ID: {hit.id}\n{label}\n{distance:.2f}")
                else:
                    axes[i].set_title("❌ Bild nicht gefunden")
                    axes[i].axis('off')

            plt.suptitle(f"🔍 Ähnliche Bilder zu: {query_image}")
            plt.tight_layout()
            plt.show()
        else:
            for hit in results[0]:
                print(f"ID: {hit.id}, Label: {hit.entity.get('label')}, Distance: {hit.distance:.2f}")

        return results

if __name__ == "__main__":
    # Beispiel-Parameter: Pfad zum Abfragebild und Top-K Ergebnisse
    QUERY_IMAGE = r"C:\Users\jfham\OneDrive\Desktop\milvus_local\images\Images\n02106662-German_shepherd\n02106662_662.jpg"
    TOP_K = 10

    searcher = ImageSearcher()
    searcher.search(QUERY_IMAGE, top_k=TOP_K)
