import os
import torch
import open_clip
from PIL import Image
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class CLIPImageIndexer:
    def __init__(
        self,
        image_dir="images_v3",
        collection_name="image_vectors",
        dim=768,  # Ensure that the embedding dimension matches the chosen model (e.g., 768 for ViT-L-14).
        db_batch_size=4096,   # Batch size for inserting into the DB
        model_batch_size=128,  # Batch size for model inference
        host="localhost",
        port="19530"
    ):
        """
        Initializes a CLIPImageIndexer instance.

        :param image_dir: Directory containing the images to be indexed.
        :param collection_name: Name of the Milvus collection where embeddings are stored.
        :param dim: Dimension of the OpenCLIP embeddings (e.g., 768 for ViT-L-14).
        :param db_batch_size: Number of embeddings per batch to be inserted into Milvus.
        :param model_batch_size: Number of images to be processed simultaneously by the model.
        :param host: Hostname for the Milvus connection.
        :param port: Port for the Milvus connection.
        """
        self.image_dir = image_dir
        self.collection_name = collection_name
        self.dim = dim
        self.db_batch_size = db_batch_size
        self.model_batch_size = model_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delete_existing = False  # If True, an existing collection with the same name will be dropped.

        self._setup_model()
        self._connect_milvus(host, port)
        self._prepare_collection()

    def _setup_model(self):
        """
        Loads the ViT-L-14 model with pretrained weights in FP16 precision to reduce memory usage.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="openai",  # Use pretrained weights from OpenAI
            precision='fp16',      # FP16 precision
            device=self.device,
            force_quick_gelu=True,
            jit=True
        )
        self.model.eval()
 
    def _connect_milvus(self, host, port):
        """
        Establishes a connection to the Milvus server using the provided host and port.
        """
        connections.connect("default", host=host, port=port)

    def _prepare_collection(self):
        """
        Creates (or rebuilds if delete_existing=True) the Milvus collection with the appropriate schema and indexes.
        """
        # If a collection with the same name exists and delete_existing=True, drop it.
        if self.collection_name in utility.list_collections() and self.delete_existing:
            Collection(self.collection_name).drop()
            print(f"🗑️  Existing collection '{self.collection_name}' was dropped.")

        # Define the schema fields.
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255)  # Stores the image path
        ]
        schema = CollectionSchema(fields, description="OpenCLIP Image Embeddings with Paths")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create an index for the embedding field.
        self.collection.create_index(
            "embedding",
            {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
        )

        # Try to create an index for the 'path' field; if it fails, use a fallback.
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
            print(f"⚠️ Error creating Trie index for 'path': {e}")
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
                print(f"⚠️ Error creating fallback index for 'path': {fallback_e}")

        # Load the collection into memory for search.
        self.collection.load()

    def _batch_image_to_vector(self, image_paths):
        """
        Converts a list of image paths into a batch of normalized OpenCLIP embedding vectors.
        Images that cannot be loaded are skipped.
        
        :param image_paths: List of paths to the images.
        :return: Tuple (Tensor of embeddings, List of valid image paths)
        """
        images = []
        valid_paths = []  # Keep track of images that were successfully loaded
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.preprocess(img))
                valid_paths.append(path)
            except Exception as e:
                print(f"⚠️ Error opening image {path}: {e}")
        
        if not images:
            raise ValueError("⚠️ No valid image found in the current batch.")
        
        # Stack images into a batch tensor and move to the specified device.
        image_tensor = torch.stack(images).to(self.device).half()
        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensor, normalize=True)  # Shape: [N, dim]
            
        return embeddings, valid_paths

    def _generate_embeddings(self):
        db_batch_ids = []
        db_batch_vecs = []
        db_batch_paths = []
        model_batch_paths = []
        current_idx = 0

        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)
                    escaped_path = image_path.replace("\\", "\\\\")
                    query_expr = f'path in ["{escaped_path}"]'
                    try:
                        query_result = self.collection.query(
                            expr=query_expr,
                            output_fields=["path"]
                        )
                    except Exception as e:
                        print(f"⚠️ Milvus query error for {image_path}: {e}")
                        query_result = []

                    if query_result:
                        print(f"⚠️ Image already indexed: {image_path}")
                        continue

                    model_batch_paths.append(image_path)

                    if len(model_batch_paths) == self.model_batch_size:
                        try:
                            embeddings, valid_paths = self._batch_image_to_vector(model_batch_paths)
                            for i, embedding in enumerate(embeddings):
                                db_batch_ids.append(current_idx)
                                db_batch_vecs.append(embedding)
                                db_batch_paths.append(valid_paths[i])
                                print(f"✅ [{current_idx}] {os.path.basename(valid_paths[i])} indexed.")
                                current_idx += 1
                        except Exception as e:
                            print(f"⚠️ Error processing model batch {model_batch_paths}: {e}")
                        model_batch_paths = []
                        
                        if len(db_batch_ids) >= self.db_batch_size:
                            yield db_batch_ids, db_batch_vecs, db_batch_paths
                            db_batch_ids, db_batch_vecs, db_batch_paths = [], [], []

        if model_batch_paths:
            try:
                embeddings, valid_paths = self._batch_image_to_vector(model_batch_paths)
                for i, embedding in enumerate(embeddings):
                    db_batch_ids.append(current_idx)
                    db_batch_vecs.append(embedding)
                    db_batch_paths.append(valid_paths[i])
                    print(f"✅ [{current_idx}] {os.path.basename(valid_paths[i])} indexed.")
                    current_idx += 1
            except Exception as e:
                print(f"⚠️ Error processing final model batch {model_batch_paths}: {e}")

        if db_batch_ids:
            yield db_batch_ids, db_batch_vecs, db_batch_paths

    def index_images(self):
        """
        Iterates through the image directory and inserts all computed embeddings into Milvus in batches.
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
