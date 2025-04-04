from pymilvus import connections, Collection

# Verbindung zu Milvus herstellen
connections.connect("default", host="localhost", port="19530")
collection = Collection("image_vectors")
collection.load()

# Beispielabfrage: Zeige alle Einträge mit id zwischen 0 und 10
expr = "id >= 0 and id < 10"
results = collection.query(expr=expr, output_fields=["id", "embedding", "label"])

print("Einträge in der Milvus Collection:")
for entry in results:
    print(f"ID: {entry['id']}")
    print(f"Label: {entry['label']}")
    # Die Embedding-Vektoren sind oft sehr lang (z. B. 2048 Dimensionen)
    print(f"Embedding (erste 10 Werte): {entry['embedding'][:10]} ...\n")
