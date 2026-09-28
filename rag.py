import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# Load Sephora product dataset
df = pd.read_csv("Dataset/product_info.csv")

# Columns useful for product retrieval
rag_columns = [
    "product_id",
    "product_name",
    "brand_name",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "price_usd",
    "rating",
    "reviews",
    "loves_count",
    "highlights",
    "ingredients"
]

products = df[rag_columns].copy()

print("Number of products:", len(products))
print(products.head())

# Convert missing values safely into text


def safe_text(value):
    if pd.isna(value):
        return "Not available"
    return str(value)


# Convert each product row into a text document for RAG
def create_product_document(row):
    return f"""
Product: {safe_text(row['product_name'])}
Brand: {safe_text(row['brand_name'])}
Category: {safe_text(row['primary_category'])}
Subcategory: {safe_text(row['secondary_category'])}
Product Type: {safe_text(row['tertiary_category'])}
Price: ${safe_text(row['price_usd'])}
Rating: {safe_text(row['rating'])}
Reviews: {safe_text(row['reviews'])}
Customer Loves: {safe_text(row['loves_count'])}
Highlights: {safe_text(row['highlights'])}
Ingredients: {safe_text(row['ingredients'])[:1000]}
""".strip()


products["document"] = products.apply(
    create_product_document,
    axis=1
)

print("\nExample product document:\n")
print(products["document"].iloc[0])

# ---------------------------------------------------------
# Generate embeddings for product documents
# ---------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = products["document"].tolist()

print("Generating embeddings...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbedding shape:", embeddings.shape)

# ---------------------------------------------------------
# Build FAISS vector index
# ---------------------------------------------------------

# FAISS expects float32 vectors
embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

# Each embedding has 384 dimensions
dimension = embeddings.shape[1]

# Create FAISS index
# Since embeddings are normalized, inner product acts as cosine similarity
index = faiss.IndexFlatIP(dimension)

# Add all product embeddings to the index
index.add(embeddings)
faiss.write_index(index, "sephora_products.index")

print("\nFAISS index created successfully.")
print("Embedding dimension:", dimension)
print("Products stored in FAISS:", index.ntotal)
