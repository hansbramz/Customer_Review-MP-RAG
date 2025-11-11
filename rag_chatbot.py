import streamlit as st
import json
import time
import requests
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

# 1. Aggregated Data Source (Live Fetch) - ENHANCED VERSION
def fetch_product_data_from_mysql() -> List[Dict]:
    """
    Connects to MySQL, aggregates review data by product, and returns RAG documents.
    Returns an empty list and displays an error if connection or query fails.
    """
    
    engine = initialize_mysql_engine()
    if not engine:
        st.error("MySQL connection initialization failed. Cannot proceed without data.")
        return [] 
        
    st.sidebar.info("🔄 Fetching ALL product data from MySQL...")
    
    # SQL Aggregation Query - Fetch ALL products without LIMIT
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
                GROUP_CONCAT(review_text ORDER BY id SEPARATOR ' | '),
                1, 500
            ) AS review_snippet
        FROM {TABLE_NAME}
        WHERE product_name IS NOT NULL 
        AND product_id IS NOT NULL
        GROUP BY product_id, product_name, Brand
        HAVING COUNT(id) > 0
        ORDER BY review_count DESC
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
            
            # Handle NULL values
            product_name = product_name or "Unknown Product"
            brand = brand or "Unknown Brand"
            emotions = emotions or "No emotion data"
            review_snippet = review_snippet or "No reviews available"
            
            # --- Format into RAG Document Structure ---
            context_data = {
                "id": str(product_id),  # Use product_id as string for consistency
                "name": product_name,
                "sku": product_id, 
                "brand": brand,
                "description": f"Brand: {brand}. Total Reviews: {review_count}. Review Snippet: {review_snippet}",
                "marketplace_names": [product_name, brand, product_id],  # Include all searchable names
                "avg_rating": round(float(avg_rating), 2) if avg_rating else 0.0,
                "sentiment_score": round(float(sentiment_score), 2) if sentiment_score else 0.0,
                "review_count": review_count,
                "emotion_summary": f"Key customer emotions: {emotions}. Based on {review_count} reviews.",
                "full_context": f"{product_name} by {brand} (SKU: {product_id})"  # For search
            }
            aggregated_products.append(context_data)
        
        if len(aggregated_products) == 0:
            st.sidebar.warning("⚠️ No products found in database. Check your table data.")
        else:
            st.sidebar.success(f"✅ Successfully loaded {len(aggregated_products)} products from MySQL.")
        
        return aggregated_products
        
    except SQLAlchemyError as e:
        st.error(f"❌ Error querying/aggregating data from MySQL: {e}")
        return [] 
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during database processing: {e}")
        return []


# --- RAG Core Functions (The "Brain") ---

# 2. Vector Index Initialization and Ingestion - ENHANCED
def create_product_knowledge_base(data: List[Dict], embeddings_model) -> Optional[List[Dict]]:
    """
    Creates the knowledge base by combining product data into embeddable context chunks.
    """
    if not data:
        st.error("No product data available to build knowledge base.")
        return []
    
    if not embeddings_model:
        st.error("Embedding model not ready. Cannot build knowledge base.")
        return None
    
    knowledge_base = []
    texts_to_embed = []
    
    # Combine all relevant fields into a single context string (the "chunk")
    for item in data:
        context = (
            f"Product Name: {item['name']}. "
            f"SKU/Product ID: {item['sku']}. "
            f"Brand: {item['brand']}. "
            f"Description: {item['description']} "
            f"Also known as: {', '.join(item.get('marketplace_names', []))}. "
            f"Average Customer Rating: {item['avg_rating']}/5.0. "
            f"Total Reviews: {item['review_count']}. "
            f"Sentiment Score: {item['sentiment_score']}. "
            f"{item['emotion_summary']}"
        )
        texts_to_embed.append(context)
        knowledge_base.append({
            "id": item['id'], 
            "name": item['name'],
            "brand": item['brand'],
            "sku": item['sku'],
            "context": context,
            "searchable_text": item['full_context'].lower(),
            "product_data": item  # Store full product data for reference
        })
        

    st.sidebar.info(f"🔨 Building searchable index for {len(data)} products...")
    
    # --- Mocking the Embedding Process ---
    for i, text in enumerate(texts_to_embed):
        # In production: response = embeddings_model.embed_content(model="text-embedding-004", content=text)
        mock_vector = [len(text) + i] * 5 
        knowledge_base[i]['embedding'] = mock_vector
        
    st.sidebar.success(f"✅ Knowledge Base ready with {len(knowledge_base)} documents.")
    return knowledge_base

# 3. ENHANCED Retrieval Function (Better Similarity Search)
def retrieve_relevant_context(query: str, knowledge_base: List[Dict], top_k: int = 3) -> str:
    """
    Enhanced similarity search that properly scores and ranks all products.
    """
    if not knowledge_base:
        return "No relevant product information found in the knowledge base."
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_documents = []
    
    for doc in knowledge_base:
        score = 0
        context_lower = doc['context'].lower()
        product_data = doc['product_data']
        
        # Scoring system (more sophisticated)
        
        # 1. Exact SKU/Product ID match (highest priority)
        if doc['sku'].lower() in query_lower:
            score += 100
        
        # 2. Exact product name match
        if doc['name'].lower() in query_lower:
            score += 50
        
        # 3. Brand name match
        if doc['brand'].lower() in query_lower:
            score += 30
        
        # 4. Partial name match (word-level)
        name_words = set(doc['name'].lower().split())
        matching_words = query_words.intersection(name_words)
        score += len(matching_words) * 10
        
        # 5. Marketplace name aliases
        for alias in product_data.get('marketplace_names', []):
            if alias.lower() in query_lower:
                score += 20
        
        # 6. Context keyword match
        context_words = set(context_lower.split())
        context_matches = query_words.intersection(context_words)
        score += len(context_matches) * 2
        
        # 7. Sentiment/emotion keywords
        if any(word in query_lower for word in ['review', 'rating', 'sentiment', 'emotion', 'feedback']):
            score += 5
        
        # Only include documents with some relevance
        if score > 0:
            scored_documents.append({
                'doc': doc,
                'score': score,
                'name': doc['name'],
                'brand': doc['brand']
            })
    
    # Sort by score (highest first)
    scored_documents.sort(key=lambda x: x['score'], reverse=True)
    
    if not scored_documents:
        return f"No relevant product information found for query: '{query}'. Try searching by product name, brand, or SKU."
    
    # Get top K results
    top_results = scored_documents[:top_k]
    
    # Format context for LLM
    context_chunks = []
    for i, result in enumerate(top_results, 1):
        context_chunks.append(
            f"[Product {i}] {result['doc']['context']} "
            f"(Relevance Score: {result['score']})"
        )
    
    formatted_context = "\n\n---\n\n".join(context_chunks)
    
    # Add summary header
    found_products = ", ".join([f"'{r['name']}' by {r['brand']}" for r in top_results[:3]])
    header = f"Found {len(scored_documents)} matching products. Top matches: {found_products}\n\n"
    
    return f"{header}Product Knowledge Context (Retrieved via Similarity Search):\n\n{formatted_context}"

# 4. LLM Generation - WITH CLAUDE FALLBACK
def generate_response(query: str, context: str):
    """Calls the Gemini API (or Claude as fallback) to generate a grounded response."""
    system_prompt = (
        "You are a helpful and detailed RAG (Retrieval-Augmented Generation) Chatbot specializing in product support and information. "
        "Your goal is to answer the user's question accurately, using ONLY the provided 'Product Knowledge Context'. "
        "The context includes product details, customer ratings, sentiment analysis, and review summaries. "
        "If the context does not contain the answer, state clearly that you do not have enough information. "
        "Always be concise, professional, and friendly. "
        "When comparing products, use the data provided to give objective comparisons."
    )
    
    user_query = f"{context}\n\nUser Question: {query}"
    
    # Add throttling: wait at least 2 seconds between API calls
    if 'last_api_call' in st.session_state:
        elapsed = time.time() - st.session_state.last_api_call
        if elapsed < 2:
            time.sleep(2 - elapsed)
    
    st.session_state.last_api_call = time.time()
    
    # Try different Gemini models
    models_to_try = [
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
    ]
    
    for model_name in models_to_try:
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apiKey}"

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }

        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(apiUrl, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not parse model response.')
                return text
                
            elif response.status_code == 429:
                print(f"Rate limit hit on {model_name}, trying next model...")
                time.sleep(3)
                continue
            else:
                print(f"API Error on {model_name}: Status {response.status_code}")
                continue
                
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Try Claude API as fallback
    try:
        claude_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if claude_key:
            print("Trying Claude API as fallback...")
            claude_url = "https://api.anthropic.com/v1/messages"
            
            claude_payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_query}]
            }
            
            claude_headers = {
                'Content-Type': 'application/json',
                'x-api-key': claude_key,
                'anthropic-version': '2023-06-01'
            }
            
            response = requests.post(claude_url, headers=claude_headers, json=claude_payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('content', [{}])[0].get('text', 'Error: Could not parse Claude response.')
                return text
    except Exception as e:
        print(f"Claude API fallback failed: {e}")
    
    return ("⚠️ **API Rate Limit Exceeded**\n\n"
            "All available AI models are currently rate-limited. Please wait 60 seconds and try again.")

# --- Streamlit UI and Logic ---

if '__app_id' not in globals():
    __app_id = 'default-app-id'
if '__firebase_config' not in globals():
    __firebase_config = '{}'

if 'user_id' not in st.session_state:
    st.session_state.user_id = "local-dev-user"
    st.session_state.is_auth_ready = True

st.set_page_config(page_title="Product RAG Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ Product Knowledge RAG Chatbot")
st.caption("AI-powered product search and analysis using live MySQL data + Vector Search")

# Initialize knowledge base in session state (CACHE IT)
if 'knowledge_base' not in st.session_state or st.sidebar.button("🔄 Refresh Database"):
    with st.spinner("Loading product data from database..."):
        product_data = fetch_product_data_from_mysql()
        
        st.session_state.embeddings_model = True  # Mock
        
        st.session_state.knowledge_base = create_product_knowledge_base(
            product_data, 
            st.session_state.get('embeddings_model')
        )
        
        # Store raw product data for display
        st.session_state.product_data = product_data

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I can help you with product information, ratings, and customer sentiment from our database. Ask me about any product by name, brand, or SKU!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about a product..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            
            # Retrieve context
            context = retrieve_relevant_context(prompt, st.session_state.knowledge_base, top_k=3)
            
            # Show retrieved context (optional - can be hidden)
            with st.expander("🔍 Retrieved Context (Debug View)"):
                st.code(context, language='text')

            # Generate response
            with st.spinner("🤖 Generating response..."):
                response = generate_response(prompt, context)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with system info
st.sidebar.header("📊 System Status")
st.sidebar.markdown(f"**App ID:** `{__app_id}`")
st.sidebar.markdown(f"**User:** `{st.session_state.user_id}`")

if st.session_state.get('knowledge_base'):
    kb_count = len(st.session_state.knowledge_base)
    st.sidebar.metric("Products in Knowledge Base", kb_count)
    
    if kb_count > 0:
        st.sidebar.success("✅ Database Connected")
    else:
        st.sidebar.error("❌ No products loaded")
else:
    st.sidebar.warning("⚠️ Knowledge base not initialized")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Try asking:**")
st.sidebar.markdown("- Tell me about [product name]")
st.sidebar.markdown("- What's the rating for [product]?")
st.sidebar.markdown("- Compare [product A] and [product B]")
st.sidebar.markdown("- Show products from [brand]")
st.sidebar.markdown("- What do customers say about [product]?")

# Optional: Show sample products
if st.sidebar.checkbox("Show Sample Products"):
    if st.session_state.get('product_data'):
        st.sidebar.markdown("### Sample Products:")
        for i, product in enumerate(st.session_state.product_data[:5], 1):
            st.sidebar.markdown(f"{i}. **{product['name']}** by {product['brand']} ({product['review_count']} reviews)")
