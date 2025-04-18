import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround for OpenMP errors

import torch
import open_clip
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
        Loads the ViT-L-14 model with OpenAI pretrained weights using FP16 precision.
        This ensures the same embedding space as used during indexing.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",     # Consistent with the indexer code
            pretrained="openai",       # OpenAI pretrained weights
            precision='fp16',          # FP16 precision for memory efficiency
            device=self.device,
            force_quick_gelu=True,
            jit=True
        )
        self.model.eval()

    def _connect_milvus(self, host, port):
        """
        Connects to the Milvus server.

        :param host: Host name for the Milvus server.
        :param port: Port for the Milvus server.
        """
        connections.connect("default", host=host, port=port)

    def _image_to_vector(self, image_path):
        """
        Converts an image to a normalized OpenCLIP embedding vector.

        :param image_path: Path to the image.
        :return: Normalized embedding vector (as a PyTorch tensor).
        """
        img = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            features = self.model.encode_image(image_input, normalize=True)
        return features.flatten()


    def _text_to_vector(self, query_text):
        """
        Converts text into a normalized CLIP text embedding vector.

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
        Searches the Milvus collection using the provided query.
        The query can either be an image file path or a text string.

        :param query: An image file path or a text string.
        :param top_k: Number of top results to return.
        :param visualize: If True, displays the top_k results.
        :return: Search results.
        """
        if os.path.exists(query):
            # Image-based search
            print(f"🔍 Searching with image: {query}")
            query_vector = self._image_to_vector(query).tolist()
        else:
            # Text-based search
            print(f"🔍 Searching with text: {query}")
            query_vector = self._text_to_vector(query).tolist()

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
    # Example: either image or text query. Make sure the file path or text is valid.
    QUERY = r"C:\Users\jfham\OneDrive\Desktop\milvus_local\images_v3\image_data\Fruits_Vegetables\test\cauliflower\Image_6.jpg"
    TOP_K = 5

    searcher = CLIPImageSearcher()
    searcher.search(QUERY, top_k=TOP_K)
  