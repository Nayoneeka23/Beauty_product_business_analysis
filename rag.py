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
# 6. Retrieve relevant products + apply filters
# ---------------------------------------------------------

def retrieve_products(
    query,
    max_price=None,
    min_rating=None,
    brand=None,
    final_k=5,
    candidate_k=200
):

    # Convert user question into embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Retrieve a larger semantic candidate pool
    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    results = products.iloc[
        indices[0]
    ].copy()

    results["similarity_score"] = scores[0]

    # -----------------------------------------------------
    # Brand filter
    # -----------------------------------------------------

    if brand is not None:

        results = results[
            results["brand_name"]
            .fillna("")
            .str.strip()
            .str.lower()
            == brand.strip().lower()
        ]

    # -----------------------------------------------------
    # Maximum price filter
    # -----------------------------------------------------

    if max_price is not None:

        results = results[
            results["price_usd"].notna()
            & (results["price_usd"] <= max_price)
        ]

    # -----------------------------------------------------
    # Minimum rating filter
    # -----------------------------------------------------

    if min_rating is not None:

        results = results[
            results["rating"].notna()
            & (results["rating"] >= min_rating)
        ]

    # Keep best semantic matches
    results = results.head(final_k)

    return results


# ---------------------------------------------------------
# 7. Generate answer using Mistral
# ---------------------------------------------------------

def generate_answer(
    query,
    retrieved_products,
    max_price=None,
    min_rating=None,
    brand=None
):

    context = "\n\n---\n\n".join(
        retrieved_products["document"].tolist()
    )

    # Build description of active filters
    filter_text = []

    if brand is not None:
        filter_text.append(
            f"Brand: {brand}"
        )

    if max_price is not None:
        filter_text.append(
            f"Maximum price: ${max_price:.2f}"
        )

    if min_rating is not None:
        filter_text.append(
            f"Minimum rating: {min_rating}"
        )

    if filter_text:
        filters = "\n".join(filter_text)
    else:
        filters = "No additional filters."

    prompt = f"""
You are a beauty product recommendation assistant.

Answer the user's question using ONLY the Sephora product information
provided in the context below.

Important rules:
- Recommend only products present in the provided context.
- Do not invent products.
- Do not invent prices.
- Do not invent ratings.
- Do not invent ingredients.
- Do not invent product benefits.
- Respect all active filters.
- If information is marked "Not available", do not invent it.
- Explain briefly why each recommendation matches the user's request.
- If the available information is insufficient, clearly say so.
- Keep the answer concise and easy to understand.

USER QUESTION:
{query}

ACTIVE FILTERS:
{filters}

SEPHORA PRODUCT CONTEXT:
{context}

ANSWER:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


# ---------------------------------------------------------
# 8. Helper function for optional numeric filters
# ---------------------------------------------------------

def get_optional_number(
    prompt,
    min_value=None,
    max_value=None
):

    while True:

        value = input(prompt).strip()

        # Empty input = no filter
        if value == "":
            return None

        try:
            number = float(value)

            if (
                min_value is not None
                and number < min_value
            ):
                print(
                    f"Please enter a value of at least "
                    f"{min_value}."
                )
                continue

            if (
                max_value is not None
                and number > max_value
            ):
                print(
                    f"Please enter a value no greater "
                    f"than {max_value}."
                )
                continue

            return number

        except ValueError:

            print(
                "Please enter a valid number "
                "or press Enter to skip."
            )


# ---------------------------------------------------------
# 9. Helper function for brand selection
# ---------------------------------------------------------

def get_brand():

    value = input(
        "Brand (press Enter for any brand): "
    ).strip()

    # No brand filter
    if value == "":
        return None

    # Find exact brand ignoring capitalization
    matching_brands = products[
        products["brand_name"]
        .fillna("")
        .str.lower()
        == value.lower()
    ]["brand_name"].dropna().unique()

    if len(matching_brands) > 0:
        return matching_brands[0]

    # Brand not found
    print(
        f"\nBrand '{value}' was not found "
        "in the Sephora dataset."
    )

    print(
        "Continuing without a brand filter."
    )

    return None


# ---------------------------------------------------------
# 10. Interactive RAG system
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SEPHORA AI PRODUCT ASSISTANT")
print("=" * 60)

print("""
Ask a question about Sephora products.

Examples:
- I have dry skin and need a moisturizer
- Recommend a cleanser for sensitive skin
- Suggest something for damaged hair
- I have an oily scalp and need a shampoo

Optional filters:
- Brand
- Maximum price
- Minimum rating

Press Enter to skip any filter.

Type 'exit' to stop.
""")


while True:

    # -----------------------------------------------------
    # User question
    # -----------------------------------------------------

    query = input(
        "\nAsk a Sephora product question:\n> "
    ).strip()

    if query.lower() in [
        "exit",
        "quit",
        "q"
    ]:
        print("\nGoodbye!")
        break

    if not query:
        print(
            "\nPlease enter a question."
        )
        continue

    # -----------------------------------------------------
    # Optional filters
    # -----------------------------------------------------

    print(
        "\nOptional filters "
        "(press Enter to skip):"
    )

    brand = get_brand()

    max_price = get_optional_number(
        "Maximum price ($): ",
        min_value=0
    )

    min_rating = get_optional_number(
        "Minimum rating (0-5): ",
        min_value=0,
        max_value=5
    )

    # -----------------------------------------------------
    # Retrieve + filter
    # -----------------------------------------------------

    print(
        "\nSearching Sephora products..."
    )

    results = retrieve_products(
        query=query,
        max_price=max_price,
        min_rating=min_rating,
        brand=brand,
        final_k=5,
        candidate_k=200
    )

    # -----------------------------------------------------
    # No matches
    # -----------------------------------------------------

    if results.empty:

        print(
            "\nNo suitable products were found "
            "for those search conditions."
        )

        print(
            "Try changing the brand, increasing "
            "the maximum price, lowering the "
            "minimum rating, or changing your query."
        )

        continue

    # -----------------------------------------------------
    # Display retrieved products
    # -----------------------------------------------------

    print("\nTop retrieved products:\n")

    display_columns = [
        "product_name",
        "brand_name",
        "primary_category",
        "secondary_category",
        "price_usd",
        "rating",
        "similarity_score"
    ]

    print(
        results[
            display_columns
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Generate answer using Mistral
    # -----------------------------------------------------

    print(
        "\nGenerating AI recommendation...\n"
    )

    try:

        answer = generate_answer(
            query=query,
            retrieved_products=results,
            max_price=max_price,
            min_rating=min_rating,
            brand=brand
        )

        print("=" * 60)
        print("SEPHORA AI RESPONSE")
        print("=" * 60)

        print("\n" + answer)

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        print(
            "\nERROR: Mistral took too long to respond."
        )

    except requests.exceptions.RequestException as error:

        print(
            "\nERROR while communicating with Ollama:",
            error
        )
