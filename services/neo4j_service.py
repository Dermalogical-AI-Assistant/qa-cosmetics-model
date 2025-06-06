import ast
from collections import defaultdict
from neo4j import GraphDatabase
from typing import List, Tuple
from utils.get_env_variables import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from utils.get_sentence_models import embedding_model_node_retrieval, embedding_model_sentence_retrieval

print(f'NEO4J_URI={NEO4J_URI}')
print(f'NEO4J_USERNAME={NEO4J_USERNAME}')
print(f'NEO4J_PASSWORD={NEO4J_PASSWORD}')

driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=3600,
    connection_timeout=30
)

def vector_search_by_ingredient_benefit(query_benefit: str, top_k: int = 5) -> List[Tuple[str, str]]:
    query_embedding = embedding_model_sentence_retrieval.embed_query(query_benefit)

    query = """
        CALL db.index.vector.queryNodes('product_ingredient_benefits_embedding_vector_index', $top_k, $query_embedding)
        YIELD node, score
        MATCH (node:Product)
        RETURN node.title AS title, node.ingredient_benefits AS ingredient_benefits, score
        ORDER BY score DESC
    """
    
    with driver.session() as session:
        result = session.run(query, query_embedding=query_embedding, top_k=top_k)
        products_with_ingredient_benefit = [
            { 
                'title': record["title"], 
                'ingredient_benefits': record["ingredient_benefits"] 
            } 
            for record in result
        ]
    return products_with_ingredient_benefit

def vector_search_by_description(label: str, query: str, top_k: int = 7):
    embedding = embedding_model_sentence_retrieval.embed_query(query)

    cypher = f"""
        CALL db.index.vector.queryNodes('{label.lower()}_description_embedding_vector_index', $topK, $embedding)
        YIELD node, score
        RETURN node.title AS title, score
        ORDER BY score DESC
    """

    with driver.session() as session:
        results = session.run(cypher, embedding=embedding, topK=top_k)
        return results.data()

def vector_search(label: str, query: str, top_k: int = 1):
    embedding = embedding_model_node_retrieval.embed_query(query)

    cypher = f"""
        CALL db.index.vector.queryNodes('{label.lower()}_title_vector_index', $topK, $embedding)
        YIELD node, score
        RETURN node.title AS title, score
        ORDER BY score DESC
    """

    with driver.session() as session:
        results = session.run(cypher, embedding=embedding, topK=top_k)
        return results.data()
 
def search_node_info(node_type, list_titles):
    properties = {
        'Product': [
            'id', 'title', 'description', 'how_to_use', 'ingredient_benefits', 
            'skincare_concern', 'natural', 'ewg', 'analysis_text', 'price'
        ],
        'Ingredient': ['title', 'cir_rating', 'categories', 'properties', 'integer_properties', 'introtext']
    }
    
    query = f"""
        MATCH (n:{node_type})
        WHERE n.title IN $list_titles
        RETURN n
    """
    with driver.session() as session:
        try:
            result = session.run(query, list_titles=list_titles)
            return [{k: v for k, v in dict(record["n"]).items() if k in properties[node_type]} for record in result]
        except Exception as e:
            print(f"Error executing query: {e}")
            return []

def serialize_relationship(rel):
    return {
        "rel_type": rel.type,
        "title": rel.get("title"),
        "type": rel.get("type"),
        "description": rel.get("description"),
    }

def search_triplet_info(list_products, list_ingredients, relationships):
    all_queries = []

    for r in relationships:
        conditions = []
        if list_products:
            conditions.append("p.title IN $list_products")
        if list_ingredients:
            conditions.append("i.title IN $list_ingredients")

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query_per_relationship = f"""
            MATCH (p:Product)-[r:{r}]->(i:Ingredient)
            {where_clause}
            RETURN p.title AS product_title, r, i.title AS ingredient_title
        """
        all_queries.append(query_per_relationship.strip())

    query = "\nUNION\n".join(all_queries)

    with driver.session() as session:
        try:
            result = session.run(
                query,
                list_products=list_products,
                list_ingredients=list_ingredients
            )
            
            product_titles = set()
            response = []

            for record in result:
                product_title = record["product_title"]

                # Add only the first 10 unique product titles
                if product_title not in product_titles:
                    if len(product_titles) >= 10:
                        continue
                    product_titles.add(product_title)

                # Serialize data
                record_data = {
                    "product": product_title,
                    "relationship": serialize_relationship(record["r"]),
                    "ingredient": record["ingredient_title"]
                }
                response.append(record_data)

            return response

        except Exception as e:
            print(f"Error executing query: {e}")
            return []
        
def retrieve_graph_database_without_entities(node_types, query):
    result = {}
    for node_type in node_types:
        nodes = vector_search_by_description(label=node_type,query=query, top_k=10)
        result[f"{node_type.lower()}"] = [n['title'] for n in nodes]
    
    return result

def retrieve_graph_database(list_products, list_ingredients, relationships):
    product_nodes = search_node_info(node_type='Product', list_titles=list_products)  
    ingredient_nodes = search_node_info(node_type='Ingredient', list_titles=list_ingredients)    
      
    subgraphs = []
    if relationships:
        subgraphs = search_triplet_info(
            list_products=list_products, 
            list_ingredients=list_ingredients, 
            relationships=relationships
        )
    
    # find in ingredient_benefits field in Product
    products_with_ingredient_benefit = []
    if list_ingredients:
        extracted_ingredients_str = ', '.join(list_ingredients)
        products_with_ingredient_benefit = vector_search_by_ingredient_benefit(query_benefit=extracted_ingredients_str, top_k=5)
        products_with_ingredient_benefit = [p for p in products_with_ingredient_benefit if p['title'] not in list_products]
    
    return {
        'product_nodes': product_nodes,
        'ingredient_nodes': ingredient_nodes,
        'subgraphs': subgraphs,
        'products_with_ingredient_benefit': products_with_ingredient_benefit
    }  

def retrieve_context(entities, relationships=[], node_types=[], query=""):
    if entities:
        list_products = entities.get('list_products')
        list_ingredients = entities.get('list_ingredients')
    else: 
        list_products = []
        list_ingredients = []

    if not list_products and not list_ingredients:
        data = retrieve_graph_database_without_entities(node_types=node_types, query=query)
        list_products = data.get('product')
        list_ingredients = data.get('ingredient')
        
    data = retrieve_graph_database(
        list_products=list_products, 
        list_ingredients=list_ingredients, 
        relationships=relationships
    )
    product_nodes = data['product_nodes']
    ingredient_nodes = data['ingredient_nodes']
    subgraphs = data['subgraphs']
    products_with_ingredient_benefit = data['products_with_ingredient_benefit']
    
    context = ""

    # product
    if product_nodes:
        context_product = "### Products:\n"
        for product in product_nodes:
            product_context = f"- Product {product['title']} targets {', '.join(product['skincare_concern'])} concerns."
            
            description = product.get('description')
            if description and isinstance(description, str):
                product_context += f" It is described as: {product['description'].strip()}."
              
            how_to_use = product.get('how_to_use')
            if how_to_use and isinstance(how_to_use, str):
                product_context += f" How to use: {product['how_to_use'].strip()}."
            
            ingredient_benefits = product.get('ingredient_benefits')
            if ingredient_benefits and isinstance(ingredient_benefits, str):
                product_context += f" Ingredient benefits include: {product['ingredient_benefits']}."
            
            context_product += product_context + "\n"

        context += context_product

    # ingredient
    if ingredient_nodes:
        context_ingredient = "\n### Ingredients:\n"
        
        for ingredient in ingredient_nodes:
            ingredient_context = f"- Ingredient {ingredient['title']}"

            if ingredient.get('preprocessed_introtext') and str(ingredient['preprocessed_introtext']).lower() != 'nan':
                ingredient_context += f" is {ingredient['preprocessed_introtext']}"

            if ingredient.get('properties') and str(ingredient['properties']).lower() != 'nan':
                try:
                    properties = ast.literal_eval(ingredient['properties'])
                    if isinstance(properties, list):
                        ingredient_context += f". Properties: {', '.join(properties)}"
                except Exception as e:
                    print(f"Error parsing properties: {e}")

            if ingredient.get('categories') and str(ingredient['categories']).lower() != 'nan':
                ingredient_context += f". Categories: {ingredient['categories']}"

            if ingredient.get('cir_rating'):
                ingredient_context += f". CIR rating: {ingredient['cir_rating']}"

            if ingredient.get('preprocessed_ewg_ingre', {}).get('decision'):
                ingredient_context += f". EWG decision: {ingredient['preprocessed_ewg_ingre']['decision']}"

            context_ingredient += ingredient_context + ".\n"

        context += context_ingredient

    # subgraph
    if subgraphs:
        context_subgraph = "\n### Subgraphs:\n"

        grouped = defaultdict(list)
        for graph in subgraphs:
            key = (graph['product'], graph['ingredient'])
            grouped[key].append(graph['relationship'])

        for (product, ingredient), relationships in grouped.items():
            subgraph_context = f"Product '{product}' has ingredient {ingredient}, which has the following effects:\n"
            for rel in relationships:
                effect = f"- {rel['rel_type'].lower()}"
                if rel['title']:
                    effect += f" {rel['title'].lower()}"
                if rel['description']:
                    effect += f": {rel['description']}"
                subgraph_context += f"{effect}\n"
            context_subgraph += subgraph_context + "\n"

        context += context_subgraph

    # products_with_ingredient_benefit
    if products_with_ingredient_benefit:
        context_products_with_ingredient_benefit = "\n### Products has ingredient benefits information:\n"
        for product_info in products_with_ingredient_benefit:
            context_product_with_ib = f"- Product {product_info['title']}: {product_info['ingredient_benefits']}"
            context_products_with_ingredient_benefit += (context_product_with_ib + "\n")
        context += context_products_with_ingredient_benefit
    
    return context      
