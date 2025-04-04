from pymilvus import connections, utility

# Verbindung herstellen
connections.connect(
    alias="default",
    host="localhost",
    port="19530"  # Standard-gRPC-Port
)

# Verbindung prüfen
if utility.has_collection("test_collection"):
    print("✅ Verbindung steht und Collection 'test_collection' existiert.")
else:
    print("✅ Verbindung steht! Aber Collection 'test_collection' wurde nicht gefunden.")
