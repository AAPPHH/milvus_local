import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround for OpenMP errors

import torch
import open_clip  # Ensure open_clip is installed (pip install open_clip_torch)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pymilvus import connections, Collection

class CLIPImageSearcher:
    def __init__(self,
                 collection_name="image_vectors",
                 host="localhost",
                 port="19530"):
        """
        Initializes a CLIPImageSearcher instance.

        :param collection_name: Name of the Milvus collection.
        :param host: Host to connect to Milvus.
        :param port: Port to connect to Milvus.
        """
        self.collection_name = collection_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_model()
        self._connect_milvus(host, port)
        
        # Load the Milvus collection
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _setup_model(self):
        """
        Loads the OpenCLIP model (EVA02-E-14-plus) which produces 1024-dimensional embeddings,
        so dass der gleiche Einbettungsraum wie beim Indexieren genutzt wird.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="EVA02-E-14-plus",        # Gleiches Modell wie beim Indexieren
            pretrained="laion2b_s9b_b144k",        # Gleiche Pretrained-Gewichte
            precision='fp16',                      # FP16 zur Speicherreduktion
            device=self.device
        )
        self.model.eval()

    def _connect_milvus(self, host, port):
        """
        Connects to the Milvus server using the specified host and port.
        """
        connections.connect("default", host=host, port=port)

    def _image_to_vector(self, image_path):
        """
        Converts an image to a normalized OpenCLIP embedding.

        :param image_path: Path to the image.
        :return: Normalized embedding vector.
        """
        img = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            features = self.model.encode_image(image_input, normalize=True)
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def _text_to_vector(self, query_text):
        """
        Converts text to a normalized OpenCLIP text embedding.

        :param query_text: Input text.
        :return: Normalized text embedding vector.
        """
        text_token = open_clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_token, normalize=True)
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def search(self, query, top_k=5, visualize=True):
        """
        Searches the Milvus collection using a query, which can be either an image path or text.

        :param query: An image file path or a text string.
        :param top_k: Number of top results to return.
        :param visualize: If True, displays the top_k results.
        :return: Search results.
        """
        if os.path.exists(query):
            # Bildbasierte Suche
            print(f"🔍 Searching with image: {query}")
            query_vector = self._image_to_vector(query).tolist()
        else:
            # Textbasierte Suche
            print(f"🔍 Searching with text: {query}")
            query_vector = self._text_to_vector(query).tolist()

        # Suche in Milvus – hier nur das "path"-Feld abfragen, da kein "label" existiert.
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "nprobe": 10},
            limit=top_k,
            output_fields=["path"]
        )

        print(f"🔍 Top {top_k} results for query: {query}")
        for hit in results[0]:
            image_path = hit.entity.get("path")
            print(f"ID: {hit.id}, Path: {image_path}, Distance: {hit.distance:.4f}")

        if visualize:
            fig, axes = plt.subplots(1, top_k, figsize=(15, 5))
            for i, hit in enumerate(results[0]):
                # Hole den Bildpfad direkt aus der Milvus Collection
                image_path = hit.entity.get("path")
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                    axes[i].axis("off")
                    axes[i].set_title(f"ID: {hit.id}\n{hit.distance:.2f}")
                else:
                    axes[i].text(0.5, 0.5, "Image not found", horizontalalignment="center")
                    axes[i].axis("off")
            plt.tight_layout()
            plt.show()

        return results

if __name__ == "__main__":
    # Beispiel einer textbasierten Suche:
    QUERY = r"C:\Users\jfham\OneDrive\Desktop\milvus_local\images_v3\image_data\coco2017_train\train2017\000000015239.jpg"  # Textabfrage
    TOP_K = 5

    searcher = CLIPImageSearcher()
    searcher.search(QUERY, top_k=TOP_K)
