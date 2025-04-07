import os
import torch
import clip  # Make sure the clip package is installed (pip install git+https://github.com/openai/CLIP.git)
from PIL import Image
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class CLIPImageIndexer:
    def __init__(self, 
                 image_dir="images_v3", 
                 collection_name="image_vectors",
                 dim=768,
                 batch_size=500,
                 host="localhost",
                 port="19530"):
        """
        Initializes a CLIPImageIndexer instance.

        :param image_dir: Directory where the images to be indexed are located.
        :param collection_name: Name of the Milvus collection where the vectors are stored.
        :param dim: Dimension of the CLIP embeddings (default: 768 for ViT-L/14).
        :param batch_size: Number of embeddings processed in a batch.
        :param host: Hostname for connecting to Milvus.
        :param port: Port for connecting to Milvus.
        """
        self.image_dir = image_dir
        self.collection_name = collection_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delete_existing = True  # Existing collection with the same name will be dropped if present.

        self._setup_model()
        self._connect_milvus(host, port)
        self._prepare_collection()

    def _setup_model(self):
        """
        Loads the CLIP model (ViT-L/14) and prepares the preprocess function.
        """
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def _connect_milvus(self, host, port):
        """
        Connects to the Milvus server using the specified host and port.
        """
        connections.connect("default", host=host, port=port)

    def _prepare_collection(self):
        """
        Creates (or rebuilds if delete_existing is True) the Milvus collection
        with the corresponding schema and index.
        """
        if self.collection_name in utility.list_collections() and self.delete_existing:
            Collection(self.collection_name).drop()
            print(f"🗑️  Existing collection '{self.collection_name}' has been dropped.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255)  # Field to store the image path
        ]
        schema = CollectionSchema(fields, description="CLIP Image Embeddings with Paths")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create index for the embedding field
        self.collection.create_index("embedding", {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        })

        # Optional: Create an index for the path field to make queries more efficient
        try:
            self.collection.create_index("path", {
                "index_type": "Trie",  # Alternatively "FLAT", depending on the Milvus version
                "params": {}
            })
            print("✅ Index for 'path' has been created.")
        except Exception as e:
            print(f"⚠️  Error creating index for 'path': {e}")

        self.collection.load()

    def _image_to_vector(self, image_path):
        """
        Generates a CLIP embedding for a single image.

        :param image_path: Path to the image.
        :return: Normalized embedding vector.
        """
        img = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def _generate_embeddings(self):
        """
        Generator that iterates over all images in image_dir and generates their embeddings.
        The embeddings are collected in batches and, once the batch size is reached,
        a tuple (batch_ids, batch_vecs, batch_paths) is returned.
        Before processing, it checks if the image path is already present in Milvus.
        """
        batch_ids = []
        batch_vecs = []
        batch_paths = []

        current_idx = 0

        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)

                    # Escape backslashes in the path so that Milvus parses the query correctly.
                    escaped_path = image_path.replace("\\", "\\\\")
                    query_expr = f'path in ["{escaped_path}"]'
                    
                    # Check if the image has already been indexed.
                    try:
                        query_result = self.collection.query(
                            expr=query_expr, 
                            output_fields=["path"]
                        )
                    except Exception as e:
                        print(f"⚠️  Milvus query error for {image_path}: {e}")
                        query_result = []

                    if query_result:
                        print(f"⚠️  Image already indexed: {image_path}")
                        continue

                    try:
                        embedding = self._image_to_vector(image_path)
                        batch_ids.append(current_idx)
                        batch_vecs.append(embedding)
                        batch_paths.append(image_path)

                        print(f"✅ [{current_idx}] {os.path.basename(image_path)} indexed.")
                        current_idx += 1

                        if len(batch_ids) == self.batch_size:
                            yield batch_ids, batch_vecs, batch_paths
                            batch_ids, batch_vecs, batch_paths = [], [], []
                    except Exception as e:
                        print(f"⚠️  Error processing {image_path}: {e}")

        if batch_ids:
            yield batch_ids, batch_vecs, batch_paths

    def index_images(self):
        """
        Iterates over the image directory and inserts all embeddings into the Milvus collection
        in batches, as provided by _generate_embeddings().
        """
        print("🔍 Searching through the image directory...")
        total_inserted = 0

        for batch_ids, batch_vecs, batch_paths in self._generate_embeddings():
            self._insert_and_flush(batch_ids, batch_vecs, batch_paths)
            total_inserted += len(batch_ids)

        print(f"\n🚀 A total of {total_inserted} new vectors have been inserted.")

    def _insert_and_flush(self, batch_ids, batch_vecs, batch_paths):
        """
        Inserts a batch of embeddings into Milvus and flushes the collection.
        """
        self.collection.insert([batch_ids, batch_vecs, batch_paths])
        self.collection.flush()
        print(f"✅ Batch inserted: {len(batch_ids)} entries.")

if __name__ == "__main__":
    indexer = CLIPImageIndexer()
    indexer.index_images()
    print("✅ All images have been successfully indexed and stored.")
