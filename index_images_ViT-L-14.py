import os
import torch
import open_clip
from PIL import Image
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class CLIPImageIndexer:
    def __init__(
        self,
        image_dir="images_v3",
        collection_name="image_vectors",
        dim=768,  # Ensure the embedding dimension is correct for the chosen model (e.g., 768 for ViT-L-14).
        batch_size=1000,
        host="localhost",
        port="19530"
    ):
        """
        Initializes a CLIPImageIndexer instance.

        :param image_dir: Directory containing images to be indexed.
        :param collection_name: Name of the Milvus collection where embeddings will be stored.
        :param dim: Dimension of OpenCLIP embeddings (e.g., 768 for ViT-L-14).
        :param batch_size: Number of embeddings per batch to insert into Milvus.
        :param host: Hostname for connecting to Milvus.
        :param port: Port for connecting to Milvus.
        """
        self.image_dir = image_dir
        self.collection_name = collection_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delete_existing = False  # If True, an existing collection with the same name will be dropped.

        self._setup_model()
        self._connect_milvus(host, port)
        self._prepare_collection()

    def _setup_model(self):
        """
        Loads the ViT-L-14 model with pretrained weights in FP16 precision to reduce memory consumption.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="openai",  # Using pretrained weights from OpenAI
            precision='fp16',      # FP16 precision
            device=self.device
        )
        self.model.eval()
 
    def _connect_milvus(self, host, port):
        """
        Connects to the Milvus server with the given host and port.
        """
        connections.connect("default", host=host, port=port)

    def _prepare_collection(self):
        """
        Creates (or rebuilds, if delete_existing=True) the Milvus collection
        with the corresponding schema and index.
        """
        # If a collection with the same name exists and delete_existing=True, drop it.
        if self.collection_name in utility.list_collections() and self.delete_existing:
            Collection(self.collection_name).drop()
            print(f"🗑️  Existing collection '{self.collection_name}' has been dropped.")

        # Define the schema fields.
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255)  # Store the image path
        ]
        schema = CollectionSchema(fields, description="OpenCLIP Image Embeddings with Paths")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create an index for the embedding field
        self.collection.create_index(
            "embedding",
            {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
        )

        # Try to create an index for the path field for quicker lookup; fall back if Trie fails
        try:
            self.collection.create_index(
                "path",
                {
                    "index_type": "Trie",
                    "params": {}
                }
            )
            print("✅ Index for 'path' (Trie) was created.")
        except Exception as e:
            print(f"⚠️  Error creating Trie index for 'path': {e}")
            print("➡️  Trying fallback index 'FLAT' for 'path'...")
            try:
                self.collection.create_index(
                    "path",
                    {
                        "index_type": "FLAT",
                        "params": {}
                    }
                )
                print("✅ Fallback 'FLAT' index for 'path' was created.")
            except Exception as fallback_e:
                print(f"⚠️  Error creating 'path' index with fallback: {fallback_e}")

        # Load the collection into memory for search
        self.collection.load()

    def _image_to_vector(self, image_path):
        """
        Generates an FP16 OpenCLIP embedding for a single image with built-in normalization.
        """
        img = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            features = self.model.encode_image(image_input, normalize=True)
        return features.cpu().numpy().flatten()

    def _generate_embeddings(self):
        """
        Generator that iterates over all images in image_dir, generating their embeddings.
        Embeddings are collected in batches. Once the batch size is reached, a tuple
        (batch_ids, batch_vecs, batch_paths) is yielded.

        Before processing, the code checks if the image path is already in Milvus.
        """
        batch_ids = []
        batch_vecs = []
        batch_paths = []
        current_idx = 0

        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)
                    # Escape backslashes in the path for Milvus queries
                    escaped_path = image_path.replace("\\", "\\\\")
                    # Check if the image has already been indexed by querying its path.
                    query_expr = f'path in ["{escaped_path}"]'
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

                        # Yield the batch if it is at the specified size
                        if len(batch_ids) == self.batch_size:
                            yield batch_ids, batch_vecs, batch_paths
                            batch_ids, batch_vecs, batch_paths = [], [], []
                    except Exception as e:
                        print(f"⚠️  Error processing {image_path}: {e}")

        # Yield any remaining items in the last batch
        if batch_ids:
            yield batch_ids, batch_vecs, batch_paths

    def index_images(self):
        """
        Iterates over the image directory and inserts all embeddings into Milvus in batches.
        """
        print("🔍 Scanning image directory...")
        total_inserted = 0

        for batch_ids, batch_vecs, batch_paths in self._generate_embeddings():
            self._insert_and_flush(batch_ids, batch_vecs, batch_paths)
            total_inserted += len(batch_ids)

        print(f"\n🚀 A total of {total_inserted} new vectors were inserted.")

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
    print("✅ All images have been successfully indexed and saved.")
