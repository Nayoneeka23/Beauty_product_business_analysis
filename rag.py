import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests


# ---------------------------------------------------------
# 1. Load Sephora product dataset
# ---------------------------------------------------------

df = pd.read_csv("Dataset/product_info.csv")

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


# ---------------------------------------------------------
# 2. Handle missing values safely
# ---------------------------------------------------------

def safe_text(value):
    if pd.isna(value):
        return "Not available"
    return str(value)


# ---------------------------------------------------------
# 3. Convert each product into a text document
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 4. Load SentenceTransformer embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 5. Load existing FAISS index
# ---------------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(
    "sephora_products.index"
)

print("FAISS index loaded successfully.")
print("Products stored in FAISS:", index.ntotal)


# ---------------------------------------------------------
# 6. Retrieve relevant products using FAISS
# ---------------------------------------------------------

def retrieve_products(query, k=5):

    # Convert user's question into an embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # FAISS expects float32 vectors
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Find the most similar products
    scores, indices = index.search(
        query_embedding,
        k
    )

    # Retrieve corresponding product information
    results = products.iloc[
        indices[0]
    ].copy()

    # Add similarity scores
    results["similarity_score"] = scores[0]

    return results


# ---------------------------------------------------------
# 7. Generate final answer using Mistral
# ---------------------------------------------------------

def generate_answer(query, retrieved_products):

    # Combine retrieved product documents
    context = "\n\n---\n\n".join(
        retrieved_products["document"].tolist()
    )

    # Prompt given to Mistral
    prompt = f"""
You are a beauty product recommendation assistant.

Your task is to answer the user's question using ONLY the Sephora
product information provided in the context below.

Important rules:
- Recommend only products present in the provided context.
- Do not invent products.
- Do not invent prices, ratings, ingredients, or product benefits.
- If a rating or other information is marked "Not available",
  do not make up a value.
- Explain briefly why each recommended product matches the user's request.
- If the context does not contain enough information to answer the
  question, clearly say so.
- Keep the answer concise and easy to understand.

USER QUESTION:
{query}

SEPHORA PRODUCT CONTEXT:
{context}

ANSWER:
"""

    # Send prompt to locally running Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    # Raise an error if Ollama request failed
    response.raise_for_status()

    # Return Mistral's generated answer
    return response.json()["response"]


# ---------------------------------------------------------
# 8. Test complete RAG pipeline
# ---------------------------------------------------------

query = (
    "I have dry skin and want something hydrating. "
    "What would you recommend?"
)

print("\nUser question:")
print(query)


# STEP 1: Retrieve products from FAISS
results = retrieve_products(
    query,
    k=5
)

print("\nRetrieved products:\n")

print(
    results[
        [
            "product_name",
            "brand_name",
            "primary_category",
            "secondary_category",
            "price_usd",
            "rating",
            "similarity_score"
        ]
    ].to_string(index=False)
)


# STEP 2: Send retrieved information to Mistral
print("\nGenerating answer with Mistral...\n")

try:

    answer = generate_answer(
        query,
        results
    )

    print("RAG Answer:\n")
    print(answer)

except requests.exceptions.ConnectionError:

    print(
        "ERROR: Could not connect to Ollama. "
        "Make sure Ollama is running."
    )

except requests.exceptions.Timeout:

    print(
        "ERROR: Mistral took too long to respond."
    )

except requests.exceptions.RequestException as error:

    print(
        "ERROR while communicating with Ollama:",
        error
    )
