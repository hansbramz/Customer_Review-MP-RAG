import streamlit as st
import json
import time
import requests  # ADD THIS - proper HTTP library
from typing import List, Dict, Optional
# --- New Database Imports ---
import pandas as pd
from sqlalchemy import create_engine, text, func 
from sqlalchemy.exc import SQLAlchemyError
# --- End New Database Imports ---
# --- Load Secrets from Streamlit (st.secrets) ---
try:
    DB_USER = st.secrets["DB_USER"]
    DB_PASS = st.secrets["DB_PASS"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_DATABASE = st.secrets["DB_DATABASE"]
    TABLE_NAME = st.secrets["TABLE_NAME"]
    apiKey = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"Configuration error: Missing key {e} in secrets.toml. Please check your secrets file.")
    st.stop()
# --- End Load Secrets ---

# --- Connection and Aggregation Functions ---

def initialize_mysql_engine():
    """Initializes and returns the SQLAlchemy engine using user's credentials."""
    try:
        connection_string = (
            f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
        )
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"ERROR: Could not initialize MySQL Engine. Error: {e}")
        return None

# 1. Aggregated Data Source (Live Fetch)
def fetch_product_data_from_mysql() -> List[Dict]:
    """
    Connects to MySQL, aggregates review data by product, and returns RAG documents.
    Returns an empty list and displays an error if connection or query fails.
    """
    
    engine = initialize_mysql_engine()
    if not engine:
        st.error("MySQL connection initialization failed. Cannot proceed without data.")
        return [] 
        
    st.sidebar.info("Attempting to connect to MySQL and aggregate data...")
    
    # SQL Aggregation Query - FIXED VERSION
    aggregation_query = text(f"""
        SELECT
            product_name,
            Brand,
            product_id,
            COUNT(id) AS review_count,
            AVG(rating) AS avg_rating,
            AVG(sentiment_score) AS aggregate_sentiment_score,
            SUBSTRING(
                GROUP_CONCAT(DISTINCT emotion_label ORDER BY emotion_label SEPARATOR ', '),
                1, 200
            ) AS unique_emotions_summary,
            SUBSTRING(
                GROUP_CONCAT(review_text SEPARATOR ' | '),
                1, 250
            ) AS review_snippet
        FROM {TABLE_NAME}
        GROUP BY product_id, product_name, Brand
        HAVING COUNT(id) > 0
    """)

    try:
        with engine.connect() as connection:
            result = connection.execute(aggregation_query).fetchall()
            
        aggregated_products = []
        for row in result:
            (
                product_name, brand, product_id, review_count, 
                avg_rating, sentiment_score, emotions, review_snippet
            ) = row
            
            # --- Format into RAG Document Structure ---
            context_data = {
                "id": hash(product_id), 
                "name": product_name,
                "sku": product_id, 
                "description": f"Brand: {brand}. Total Reviews: {review_count}. Review Snippet: {review_snippet}...",
                "marketplace_names": [product_name, brand], 
                "avg_rating": round(float(avg_rating), 2),
                "sentiment_score": round(float(sentiment_score), 2),
                "emotion_summary": f"Key customer emotions observed: {emotions}. Summary derived from {review_count} reviews."
            }
            aggregated_products.append(context_data)
            
        st.sidebar.success(f"Successfully aggregated {len(aggregated_products)} unique products from MySQL.")
        return aggregated_products
        
    except SQLAlchemyError as e:
        st.error(f"Error querying/aggregating data from MySQL: {e}. Cannot fetch live data.")
        return [] 
    except Exception as e:
        st.error(f"An unexpected error occurred during database processing: {e}. Cannot fetch live data.")
        return []


# --- RAG Core Functions (The "Brain") ---

# 2. Vector Index Initialization and Ingestion
def create_product_knowledge_base(data: List[Dict], embeddings_model) -> Optional[List[Dict]]:
    """
    Creates the knowledge base by combining product data into embeddable context chunks.
    """
    if not embeddings_model:
        st.error("Embedding model not ready. Cannot build knowledge base.")
        return None
    
    knowledge_base = []
    texts_to_embed = []
    
    # Combine all relevant fields into a single context string (the "chunk")
    for item in data:
        context = (
            f"Product Name: {item['name']}. SKU: {item['sku']}. "
            f"Description: {item['description']}. "
            f"Also known as: {', '.join(item.get('marketplace_names', []))}. "
            f"Average Customer Rating: {item['avg_rating']}. " 
            f"Aggregate Positive Sentiment Score: {item['sentiment_score']}. " 
            f"Key Customer Emotion Summary: {item['emotion_summary']}" 
        )
        texts_to_embed.append(context)
        knowledge_base.append({"id": item['id'], "name": item['name'], 'context': context})
        

    st.sidebar.info(f"Simulating Vector Embedding and Indexing for {len(data)} products...")
    
    # --- Mocking the Embedding Process ---
    for i, text in enumerate(texts_to_embed):
        mock_vector = [len(text) + i] * 5 
        knowledge_base[i]['embedding'] = mock_vector
        
    st.sidebar.success(f"Knowledge Base built with {len(knowledge_base)} documents.")
    return knowledge_base

# 3. Retrieval Function (Semantic Similarity Search)
def retrieve_relevant_context(query: str, knowledge_base: List[Dict], top_k: int = 2) -> str:
    """
    Simulates vector similarity search for product retrieval.
    The key to 'fuzzy' matching is happening here based on text embeddings.
    """
    if not knowledge_base:
        return "No relevant product information found in the knowledge base."
    
    # --- MOCK SIMILARITY SEARCH FOR DEMO ---
    query_lower = query.lower()
    scored_documents = []
    
    # Fetch current product data (to accurately check marketplace names if needed)
    current_product_data = fetch_product_data_from_mysql()
    
    for doc in knowledge_base:
        score = 0
        context_lower = doc['context'].lower()
        
        # Simple keyword presence check (simulates general semantic relevance)
        if any(word in context_lower for word in query_lower.split()):
            score = 1 
        
        # Find the corresponding product data entry to check marketplace names
        product_entry = next((p for p in current_product_data if str(p.get('id')) == str(doc['id']) or p.get('name') == doc['name']), None)
        
        # Check for marketplace name match (simulating a high score for "fuzzy" match)
        if product_entry and any(
            alias.lower() in query_lower 
            for alias in product_entry.get('marketplace_names', [])
        ):
             score = 2
             
        if score > 0:
            scored_documents.append(doc)

    # Sort and return the top results
    relevant_docs = sorted(scored_documents, key=lambda x: x['id'], reverse=False) 
    
    context_chunks = [doc['context'] for doc in relevant_docs[:top_k]]
    
    if not context_chunks:
        return "No relevant product information found in the knowledge base."

    # Format the retrieved documents for the LLM prompt
    formatted_context = "\n---\n".join(context_chunks)
    return f"Product Knowledge Context (Retrieved via Similarity Search):\n{formatted_context}"

# 4. LLM Generation - FIXED VERSION USING REQUESTS
def generate_response(query: str, context: str):
    """Calls the Gemini API to generate a grounded response using requests library."""
    # System Instruction: Guiding the LLM's persona and rules
    system_prompt = (
        "You are a helpful and detailed RAG (Retrieval-Augmented Generation) Chatbot specializing in product support and information. "
        "Your goal is to answer the user's question accurately, using ONLY the provided 'Product Knowledge Context'. "
        "The context includes product details, customer ratings, and sentiment analysis. " 
        "If the context does not contain the answer, state clearly that you do not have enough information. "
        "Always be concise, professional, and friendly."
    )
    
    # User Query combining context and question
    user_query = f"{context}\n\nUser Question: {query}"
    
    model_name = "gemini-2.0-flash-exp"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apiKey}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {
        'Content-Type': 'application/json'
    }

    # Exponential Backoff Retry Logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use requests library instead of fetch_func
            response = requests.post(
                apiUrl,
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not parse model response.')
                return text
            else:
                # Handle API error response codes
                error_msg = f"API Error (Attempt {attempt+1}): Status {response.status_code}"
                print(error_msg)
                try:
                    error_detail = response.json()
                    print(f"Error details: {error_detail}")
                    # If it's a permanent error (400, 401, 403), don't retry
                    if response.status_code in [400, 401, 403]:
                        return f"Error: API request failed - {error_detail.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
                    
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"Error: Failed to get response from LLM after {max_retries} attempts. Last status: {response.status_code}"
                    
        except requests.exceptions.Timeout:
            print(f"Timeout Error (Attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Error: Request timed out after {max_retries} attempts."
                
        except requests.exceptions.RequestException as e:
            print(f"Network Error (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Error: Network failure after {max_retries} attempts. Details: {str(e)}"
                
        except Exception as e:
            print(f"Unexpected Error (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Error: Unexpected failure after {max_retries} attempts. Details: {str(e)}"
                
    return "Error: Failed to generate response."

# --- Streamlit UI and Logic ---

# Firebase and Auth Setup (Kept for environment compatibility)
if '__app_id' not in globals():
    __app_id = 'default-app-id'
if '__firebase_config' not in globals():
    __firebase_config = '{}'
# Mock user setup for local running:
if 'user_id' not in st.session_state:
    st.session_state.user_id = "local-dev-user"
    st.session_state.is_auth_ready = True

st.set_page_config(page_title="Product RAG Chatbot (MySQL/Vector Search)")
st.title("🛍️ Product Knowledge Chatbot")
st.caption("Answers grounded in **live** MySQL data (simulated via Vector Search)")

# Initialize knowledge base in session state
if 'knowledge_base' not in st.session_state:
    product_data = fetch_product_data_from_mysql()
    
    # Initialize embedding model here (using a mock/placeholder)
    st.session_state.embeddings_model = True
    
    st.session_state.knowledge_base = create_product_knowledge_base(
        product_data, 
        st.session_state.get('embeddings_model')
    )

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can provide product details, ratings, and customer sentiment, fetched live from the database. Ask me about a product!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about a product..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Start assistant response block
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            
            # 3. RAG Steps
            # A. Retrieve context using vector similarity (handles fuzzy names)
            context = retrieve_relevant_context(prompt, st.session_state.knowledge_base)
            
            # Display retrieved context in a collapsible box for debugging/transparency
            with st.expander("🔍 RAG Context Retrieved"):
                st.code(context, language='text')

            # B. Generate response
            response = generate_response(prompt, context)
            
            st.markdown(response)
            
    # 4. Update chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.header("System Status")
st.sidebar.markdown(f"**App ID:** `{__app_id}`")
st.sidebar.markdown(f"**User ID:** `{st.session_state.user_id}`")
st.sidebar.markdown(f"**Total Products in KB:** `{len(st.session_state.knowledge_base) if st.session_state.knowledge_base else 0}`")

st.sidebar.markdown("""
---
**Product Data Source**
Data is aggregated live from the MySQL table. If the product count above is 0, check the database connection.
""")
