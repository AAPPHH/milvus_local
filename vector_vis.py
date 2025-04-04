import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from pymilvus import connections, Collection
from tqdm import tqdm

# Optional: Setze die maximale Anzahl an CPUs (je nach Systemkonfiguration)
os.environ["LOKY_MAX_CPU_COUNT"] = "24"


class TSNEVisualizer:
    def __init__(self, 
                 collection_name="image_vectors",
                 batch_size=2000,
                 perplexity=30,
                 random_state=42,
                 use_interactive=True):
        """
        Initialisiert den TSNEVisualizer:
          - collection_name: Name der Milvus-Collection
          - batch_size: Größe der Batches beim Laden der Vektoren
          - perplexity: t-SNE-Parameter
          - random_state: Für Reproduzierbarkeit
          - use_interactive: Falls True, wird ein interaktiver Plot (Plotly) genutzt, sonst Matplotlib.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.perplexity = perplexity
        self.random_state = random_state
        self.use_interactive = use_interactive

        print("🔌 Verbinde zu Milvus...")
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(self.collection_name)
        self.collection.load()
        print("✅ Milvus Collection geladen.")

    def load_data(self):
        """Lädt alle Vektoren in Batches aus der Milvus-Collection."""
        print("📥 Lade Vektoren in Batches...")
        self.all_results = []
        i = 0
        # Mit tqdm visualisieren wir die Batch-Fortschritte
        while True:
            expr = f"id >= {i} and id < {i + self.batch_size}"
            batch = self.collection.query(expr=expr, output_fields=["id", "embedding", "label"])
            if not batch:
                break
            self.all_results.extend(batch)
            print(f"✅ Batch {i // self.batch_size + 1}: {len(batch)} Einträge geladen")
            i += self.batch_size
        print(f"📊 Insgesamt {len(self.all_results)} Vektoren geladen.")

    def preprocess_data(self):
        """Extrahiert Vektoren und Labels, wandelt Labels in Zahlen um und speichert die Daten."""
        if not hasattr(self, "all_results"):
            self.load_data()

        # Vektoren extrahieren
        self.vectors = np.array([entry["embedding"] for entry in self.all_results])
        # Labels extrahieren und, falls im Format "n02085620-Chihuahua", den Klassennamen übernehmen
        self.labels = [
            entry["label"].split("-")[1] if "-" in entry["label"] else entry["label"]
            for entry in self.all_results
        ]
        # Labels in Zahlen kodieren
        label_encoder = LabelEncoder()
        self.labels_num = label_encoder.fit_transform(self.labels)
        self.label_names = label_encoder.classes_
        print(f"🔢 {len(self.label_names)} Klassen erkannt: {', '.join(self.label_names)}")

    def compute_tsne(self):
        """Berechnet die t-SNE-Reduktion der Vektoren."""
        if not hasattr(self, "vectors"):
            self.preprocess_data()
        print("🧠 Berechne t-SNE... (dies kann einige Minuten dauern)")
        tsne = TSNE(n_components=2, perplexity=self.perplexity, init='pca', random_state=self.random_state)
        self.reduced = tsne.fit_transform(self.vectors)
        print("✅ t-SNE Berechnung abgeschlossen.")
        return self.reduced

    def plot(self):
        """Erstellt eine Visualisierung der t-SNE-Ergebnisse."""
        if not hasattr(self, "reduced"):
            self.compute_tsne()

        if self.use_interactive:
            try:
                import plotly.express as px
            except ImportError:
                print("⚠️ Plotly nicht installiert. Fallback auf Matplotlib.")
                self.use_interactive = False

        if self.use_interactive:
            print("🎨 Erstelle interaktiven Plot mit Plotly...")
            df = pd.DataFrame({
                "TSNE-1": self.reduced[:, 0],
                "TSNE-2": self.reduced[:, 1],
                "Label": self.labels,
                "LabelNum": self.labels_num
            })
            fig = px.scatter(
                df, x="TSNE-1", y="TSNE-2", color="Label",
                title="t-SNE Projektion der Bild-Vektoren",
                hover_data=["Label"]
            )
            fig.show()
        else:
            print("🎨 Erstelle statischen Plot mit Matplotlib...")
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(self.reduced[:, 0], self.reduced[:, 1],
                                  c=self.labels_num, cmap='tab10', alpha=0.7)
            plt.legend(*scatter.legend_elements(), title="Klassen",
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("t-SNE Projektion der Bild-Vektoren")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Parameter anpassen falls gewünscht
    visualizer = TSNEVisualizer(
        collection_name="image_vectors",
        batch_size=2000,
        perplexity=30,
        random_state=42,
        use_interactive=True  # auf False setzen, wenn du den Matplotlib-Plot bevorzugst
    )
    visualizer.load_data()
    visualizer.preprocess_data()
    visualizer.compute_tsne()
    visualizer.plot()
