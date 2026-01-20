import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path='./chroma_db')
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
col = client.get_collection('cv_figures', embedding_function=ef)
data = col.get(include=['metadatas'])

print('='*60)
print('FIGURES IN CHROMADB')
print('='*60)

for m in data['metadatas']:
    source = m.get('source_pdf', 'N/A')
    filename = m.get('filename', 'N/A')
    print(f"  {source[:40]:40} -> {filename}")

print('\n' + '='*60)
print('SEARCHING FOR VISAPP AND ALGORITHM FIGURES')
print('='*60)

found = False
for m in data['metadatas']:
    source = m.get('source_pdf', '').upper()
    filename = m.get('filename', '').upper()
    if 'VISAPP' in source or 'VISAPP' in filename or 'ALGORITHM' in source or 'ALGORITHM' in filename:
        print(f"\nFOUND:")
        print(f"  source_pdf: {m.get('source_pdf')}")
        print(f"  filename: {m.get('filename')}")
        print(f"  title: {m.get('title', 'N/A')[:50]}")
        found = True

if not found:
    print("\nNO VISAPP or ALGORITHM figures found in ChromaDB!")
    print("\nThis means they were not indexed. Check:")
    print("  1. Are they in data/figure_descriptions.json?")
    print("  2. Did you run: python src/index_figures.py?")