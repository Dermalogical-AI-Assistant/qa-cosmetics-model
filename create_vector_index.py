from typing import List, Tuple
from services.neo4j_service import driver
from utils.get_sentence_models import embedding_model_node_retrieval, embedding_model_sentence_retrieval


def fetch_titles(tx, label: str) -> List[str]:
    query = f"""
        MATCH (n:{label})
        WHERE n.title IS NOT NULL
        RETURN n.title AS title
    """
    return [record["title"] for record in tx.run(query)]


def fetch_ingredient_benefits(tx) -> List[Tuple[str, str]]:
    query = """
        MATCH (n:Product)
        WHERE n.title IS NOT NULL AND n.ingredient_benefits IS NOT NULL
        RETURN n.title AS title, n.ingredient_benefits AS benefits
    """
    results = []
    for record in tx.run(query):
        title = record["title"]
        benefits = record["benefits"]
        if benefits and isinstance(benefits, str):
            clean_benefits = benefits.replace("\n", " ").strip()
            results.append((title, clean_benefits))
    return results


def store_embeddings(tx, data: List[Tuple[str, List[float]]], field: str):
    for title, embedding in data:
        tx.run(f"""
            MATCH (n {{title: $title}})
            SET n.{field} = $embedding
        """, title=title, embedding=embedding)

def embed_and_update_titles(label: str):
    with driver.session() as session:
        print(f"🔍 Fetching {label} titles...")
        titles = session.read_transaction(fetch_titles, label)
        if not titles:
            print(f"⚠️ No titles found for {label}.")
            return

        print(f"🧠 Embedding {len(titles)} titles...")
        embeddings = embedding_model_node_retrieval.embed_documents(titles)

        print(f"💾 Storing title embeddings for {label}...")
        data = list(zip(titles, embeddings))
        session.write_transaction(store_embeddings, data, "title_embedding")
        print(f"✅ Done with title embeddings for {label}.\n")


def embed_and_update_benefits():
    with driver.session() as session:
        print(f"🔍 Fetching ingredient benefits...")
        records = session.read_transaction(fetch_ingredient_benefits)
        if not records:
            print("⚠️ No ingredient benefits found.")
            return

        titles, benefits = zip(*records)
        print(f"🧠 Embedding {len(benefits)} ingredient benefits...")
        embeddings = embedding_model_sentence_retrieval.embed_documents(benefits)

        print(f"💾 Storing benefit embeddings for Product...")
        data = list(zip(titles, embeddings))
        session.write_transaction(store_embeddings, data, "ingredient_benefits_embedding")
        print(f"✅ Done with ingredient benefit embeddings.\n")


def create_vector_index(label: str, field: str, dim: int = 768):
    index_name = f"{label.lower()}_{field}_vector_index"
    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON (n.{field})
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }} }}
        """)
        print(f"✅ Created vector index: {index_name}")


if __name__ == "__main__":
    for label in ["Product", "Ingredient"]:
        embed_and_update_titles(label)
        create_vector_index(label, "title_embedding", dim=384)

    embed_and_update_benefits()
    create_vector_index("Product", "ingredient_benefits_embedding", dim=768)

    driver.close()
