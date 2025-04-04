import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround für OpenMP-Fehler

import torch
import clip  # Stelle sicher, dass clip installiert ist
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pymilvus import connections, Collection

class CLIPImageSearcher:
    def __init__(self,
                 collection_name="image_vectors",
                 mapping_path="id_to_path.json",
                 host="localhost",
                 port="19530"):
        self.collection_name = collection_name
        self.mapping_path = mapping_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_model()
        self._connect_milvus(host, port)
        
        # Lade die Milvus-Collection und das Mapping (ID zu Bildpfad)
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r") as f:
                self.id_to_path = json.load(f)
        else:
            self.id_to_path = {}
            print(f"⚠️ Mapping-Datei '{self.mapping_path}' nicht gefunden.")

    def _setup_model(self):
        # CLIP laden – hier wird ViT-L/14 verwendet (liefert 768-dimensionale Embeddings)
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def _connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)

    def _image_to_vector(self, image_path):
        img = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def _text_to_vector(self, query_text):
        # Tokenisiere den Text und erzeuge den Textvektor
        text_token = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_token)
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def search(self, query, top_k=5, visualize=True):
        """
        Führt eine Suche in der Milvus-Collection durch.
        Wenn 'query' ein Pfad zu einer Bilddatei ist, wird ein Bildvektor genutzt.
        Falls 'query' ein Text (String) ist, wird der Textvektor verwendet.
        """
        if os.path.exists(query):
            # Bildbasierte Suche
            print(f"🔍 Suche mit Bild: {query}")
            query_vector = self._image_to_vector(query).tolist()
        else:
            # Textbasierte Suche
            print(f"🔍 Suche mit Text: {query}")
            query_vector = self._text_to_vector(query).tolist()

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "nprobe": 10},
            limit=top_k,
            output_fields=["label"]
        )

        print(f"🔍 Top {top_k} Ergebnisse für Query: {query}")
        for hit in results[0]:
            label = hit.entity.get("label")
            print(f"ID: {hit.id}, Label: {label}, Distance: {hit.distance:.4f}")

        if visualize:
            fig, axes = plt.subplots(1, top_k, figsize=(15, 5))
            for i, hit in enumerate(results[0]):
                image_path = self.id_to_path.get(str(hit.id), self.id_to_path.get(hit.id))
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                    axes[i].axis("off")
                    axes[i].set_title(f"ID: {hit.id}\n{hit.entity.get('label')}\n{hit.distance:.2f}")
                else:
                    axes[i].text(0.5, 0.5, "Bild nicht gefunden", horizontalalignment="center")
                    axes[i].axis("off")
            plt.tight_layout()
            plt.show()

        return results

if __name__ == "__main__":
    # Beispiel für textbasierte Suche:
    QUERY = r"hund mit langen haaren"
    TOP_K = 5

    searcher = CLIPImageSearcher()
    searcher.search(QUERY, top_k=TOP_K)
