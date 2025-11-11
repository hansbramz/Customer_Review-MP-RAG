import streamlit as st
import json
import time
import requests
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import create_engine, text, func 
from sqlalchemy.exc import SQLAlchemyError

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

# --- Connection Functions ---

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

# --- ENHANCED DATA FETCHING WITH ANALYTICS ---

def fetch_analytical_insights() -> Dict:
    """
    Fetches comprehensive analytical insights from the database.
    Returns aggregated statistics for deeper analysis.
    """
    engine = initialize_mysql_engine()
    if not engine:
        return {}
    
    insights = {}
    
    try:
        with engine.connect() as connection:
            # 1. Overall Statistics
            overall_query = text(f"""
                SELECT 
                    COUNT(DISTINCT product_id) as total_products,
                    COUNT(*) as total_reviews,
                    AVG(rating) as avg_rating,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_reviews,
                    COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_reviews
                FROM {TABLE_NAME}
            """)
            result = connection.execute(overall_query).fetchone()
            insights['overall'] = dict(result._mapping) if result else {}
            
            # 2. Top Complained Products (Low ratings)
            complaints_query = text(f"""
                SELECT 
                    product_name,
                    Brand,
                    COUNT(*) as complaint_count,
                    AVG(rating) as avg_rating,
                    AVG(sentiment_score) as avg_sentiment,
                    GROUP_CONCAT(DISTINCT emotion_label SEPARATOR ', ') as common_emotions
                FROM {TABLE_NAME}
                WHERE rating <= 2
                GROUP BY product_name, Brand
                ORDER BY complaint_count DESC
                LIMIT 10
            """)
            result = connection.execute(complaints_query).fetchall()
            insights['top_complaints'] = [dict(row._mapping) for row in result]
            
            # 3. Best Rated Products
            best_products_query = text(f"""
                SELECT 
                    product_name,
                    Brand,
                    COUNT(*) as review_count,
                    AVG(rating) as avg_rating,
                    AVG(sentiment_score) as avg_sentiment,
                    GROUP_CONCAT(DISTINCT emotion_label SEPARATOR ', ') as common_emotions
                FROM {TABLE_NAME}
                WHERE rating >= 4
                GROUP BY product_name, Brand
                HAVING COUNT(*) >= 3
                ORDER BY avg_sentiment DESC, avg_rating DESC
                LIMIT 10
            """)
            result = connection.execute(best_products_query).fetchall()
            insights['best_products'] = [dict(row._mapping) for row in result]
            
            # 4. Emotion Analysis
            emotion_query = text(f"""
                SELECT 
                    emotion_label,
                    COUNT(*) as count,
                    AVG(rating) as avg_rating,
                    SUBSTRING(
                        GROUP_CONCAT(DISTINCT product_name ORDER BY product_name SEPARATOR ' | '),
                        1, 200
                    ) as sample_products
                FROM {TABLE_NAME}
                WHERE emotion_label IS NOT NULL
                GROUP BY emotion_label
                ORDER BY count DESC
            """)
            result = connection.execute(emotion_query).fetchall()
            insights['emotions'] = [dict(row._mapping) for row in result]
            
            # 5. Common Complaint Keywords (from negative reviews)
            complaint_keywords_query = text(f"""
                SELECT 
                    review_text,
                    product_name,
                    rating,
                    emotion_label
                FROM {TABLE_NAME}
                WHERE rating <= 2 
                AND review_text IS NOT NULL
                ORDER BY review_date DESC
                LIMIT 50
            """)
            result = connection.execute(complaint_keywords_query).fetchall()
            insights['complaint_samples'] = [dict(row._mapping) for row in result]
            
            # 6. Sales Channel Performance
            channel_query = text(f"""
                SELECT 
                    sales_channel,
                    COUNT(*) as review_count,
                    AVG(rating) as avg_rating,
                    AVG(sentiment_score) as avg_sentiment
                FROM {TABLE_NAME}
                GROUP BY sales_channel
            """)
            result = connection.execute(channel_query).fetchall()
            insights['channel_performance'] = [dict(row._mapping) for row in result]
            
            # 7. Rating Distribution
            rating_dist_query = text(f"""
                SELECT 
                    rating,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {TABLE_NAME}), 2) as percentage
                FROM {TABLE_NAME}
                GROUP BY rating
                ORDER BY rating DESC
            """)
            result = connection.execute(rating_dist_query).fetchall()
            insights['rating_distribution'] = [dict(row._mapping) for row in result]
            
        st.sidebar.success("✅ Analytical insights loaded successfully")
        return insights
        
    except Exception as e:
        st.error(f"Error fetching analytical insights: {e}")
        return {}

def fetch_product_data_from_mysql() -> List[Dict]:
    """
    Connects to MySQL, aggregates review data by product, and returns RAG documents.
    """
    engine = initialize_mysql_engine()
    if not engine:
        st.error("MySQL connection initialization failed.")
        return [] 
        
    st.sidebar.info("🔄 Fetching product data from MySQL...")
    
    aggregation_query = text(f"""
        SELECT
            product_name,
            Brand,
            product_id,
            COUNT(id) AS review_count,
            AVG(rating) AS avg_rating,
            AVG(sentiment_score) AS aggregate_sentiment_score,
            COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_count,
            COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_count,
            SUBSTRING(
                GROUP_CONCAT(DISTINCT emotion_label ORDER BY emotion_label SEPARATOR ', '),
                1, 200
            ) AS unique_emotions_summary,
            SUBSTRING(
                GROUP_CONCAT(CASE WHEN rating <= 2 THEN review_text END SEPARATOR ' | '),
                1, 300
            ) AS negative_review_samples,
            SUBSTRING(
                GROUP_CONCAT(CASE WHEN rating >= 4 THEN review_text END SEPARATOR ' | '),
                1, 300
            ) AS positive_review_samples
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
                avg_rating, sentiment_score, negative_count, positive_count,
                emotions, negative_samples, positive_samples
            ) = row
            
            # Handle NULL values
            product_name = product_name or "Unknown Product"
            brand = brand or "Unknown Brand"
            emotions = emotions or "No emotion data"
            negative_samples = negative_samples or "No negative reviews"
            positive_samples = positive_samples or "No positive reviews"
            
            # Calculate sentiment ratio
            sentiment_ratio = f"{positive_count} positive vs {negative_count} negative"
            
            context_data = {
                "id": str(product_id),
                "name": product_name,
                "sku": product_id, 
                "brand": brand,
                "description": (
                    f"Brand: {brand}. Total Reviews: {review_count}. "
                    f"Sentiment Distribution: {sentiment_ratio}. "
                    f"Negative feedback samples: {negative_samples}. "
                    f"Positive feedback samples: {positive_samples}"
                ),
                "marketplace_names": [product_name, brand, product_id],
                "avg_rating": round(float(avg_rating), 2) if avg_rating else 0.0,
                "sentiment_score": round(float(sentiment_score), 2) if sentiment_score else 0.0,
                "review_count": review_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "emotion_summary": f"Key customer emotions: {emotions}. Based on {review_count} reviews.",
                "full_context": f"{product_name} by {brand} (SKU: {product_id})"
            }
            aggregated_products.append(context_data)
        
        if len(aggregated_products) == 0:
            st.sidebar.warning("⚠️ No products found in database.")
        else:
            st.sidebar.success(f"✅ Loaded {len(aggregated_products)} products from MySQL.")
        
        return aggregated_products
        
    except SQLAlchemyError as e:
        st.error(f"❌ Error querying data: {e}")
        return [] 

# --- ENHANCED RAG WITH ANALYTICAL CONTEXT ---

def create_analytical_context(insights: Dict) -> str:
    """
    Creates a comprehensive analytical summary for the LLM.
    """
    if not insights:
        return "No analytical insights available."
    
    context_parts = []
    
    # Overall Statistics
    if 'overall' in insights and insights['overall']:
        stats = insights['overall']
        context_parts.append(
            f"OVERALL STATISTICS:\n"
            f"- Total Products: {stats.get('total_products', 0)}\n"
            f"- Total Reviews: {stats.get('total_reviews', 0)}\n"
            f"- Average Rating: {stats.get('avg_rating', 0):.2f}/5.0\n"
            f"- Average Sentiment: {stats.get('avg_sentiment', 0):.2f}\n"
            f"- Negative Reviews: {stats.get('negative_reviews', 0)}\n"
            f"- Positive Reviews: {stats.get('positive_reviews', 0)}\n"
        )
    
    # Top Complaints
    if 'top_complaints' in insights and insights['top_complaints']:
        context_parts.append("\nTOP COMPLAINED PRODUCTS (Most Negative Reviews):")
        for i, product in enumerate(insights['top_complaints'][:5], 1):
            context_parts.append(
                f"{i}. {product['product_name']} by {product['Brand']}: "
                f"{product['complaint_count']} complaints, "
                f"avg rating {product['avg_rating']:.2f}, "
                f"emotions: {product['common_emotions']}"
            )
    
    # Best Products
    if 'best_products' in insights and insights['best_products']:
        context_parts.append("\nBEST RATED PRODUCTS (Highest Sentiment):")
        for i, product in enumerate(insights['best_products'][:5], 1):
            context_parts.append(
                f"{i}. {product['product_name']} by {product['Brand']}: "
                f"{product['review_count']} reviews, "
                f"avg rating {product['avg_rating']:.2f}, "
                f"sentiment {product['avg_sentiment']:.2f}, "
                f"emotions: {product['common_emotions']}"
            )
    
    # Emotion Analysis
    if 'emotions' in insights and insights['emotions']:
        context_parts.append("\nEMOTION ANALYSIS ACROSS ALL PRODUCTS:")
        for emotion in insights['emotions'][:8]:
            context_parts.append(
                f"- {emotion['emotion_label']}: {emotion['count']} occurrences, "
                f"avg rating {emotion['avg_rating']:.2f}"
            )
    
    # Rating Distribution
    if 'rating_distribution' in insights and insights['rating_distribution']:
        context_parts.append("\nRATING DISTRIBUTION:")
        for dist in insights['rating_distribution']:
            context_parts.append(
                f"- {dist['rating']} stars: {dist['count']} reviews ({dist['percentage']}%)"
            )
    
    return "\n".join(context_parts)

def create_product_knowledge_base(data: List[Dict], insights: Dict, embeddings_model) -> Optional[List[Dict]]:
    """
    Creates enhanced knowledge base with product data + analytical insights.
    """
    if not data:
        st.error("No product data available.")
        return []
    
    if not embeddings_model:
        st.error("Embedding model not ready.")
        return None
    
    knowledge_base = []
    
    # Add analytical summary as the first document
    analytical_summary = create_analytical_context(insights)
    knowledge_base.append({
        "id": "ANALYTICAL_SUMMARY",
        "name": "Database-Wide Analytics",
        "brand": "System",
        "sku": "ANALYTICS",
        "context": f"COMPREHENSIVE DATABASE ANALYTICS:\n\n{analytical_summary}",
        "searchable_text": "analytics insights statistics complaints best rated emotions distribution",
        "product_data": {"type": "analytics"},
        "embedding": [999] * 5  # High priority for analytics queries
    })
    
    # Add individual product documents
    for item in data:
        context = (
            f"Product Name: {item['name']}. "
            f"SKU/Product ID: {item['sku']}. "
            f"Brand: {item['brand']}. "
            f"Description: {item['description']} "
            f"Also known as: {', '.join(item.get('marketplace_names', []))}. "
            f"Average Customer Rating: {item['avg_rating']}/5.0. "
            f"Total Reviews: {item['review_count']}. "
            f"Positive Reviews: {item['positive_count']}. "
            f"Negative Reviews: {item['negative_count']}. "
            f"Sentiment Score: {item['sentiment_score']}. "
            f"{item['emotion_summary']}"
        )
        
        knowledge_base.append({
            "id": item['id'], 
            "name": item['name'],
            "brand": item['brand'],
            "sku": item['sku'],
            "context": context,
            "searchable_text": item['full_context'].lower(),
            "product_data": item,
            "embedding": [len(context)] * 5
        })
    
    st.sidebar.success(f"✅ Knowledge Base ready with {len(knowledge_base)} documents (including analytics)")
    return knowledge_base

def retrieve_relevant_context(query: str, knowledge_base: List[Dict], top_k: int = 3) -> str:
    """
    Enhanced retrieval with analytical query detection.
    """
    if not knowledge_base:
        return "No relevant information found."
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Detect if this is an analytical query
    analytical_keywords = [
        'analyze', 'analysis', 'statistics', 'compare', 'best', 'worst',
        'complaint', 'complain', 'problem', 'issue', 'most', 'least',
        'top', 'bottom', 'overall', 'summary', 'trend', 'insight',
        'emotion', 'sentiment', 'rating', 'distribution', 'percentage'
    ]
    
    is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
    
    scored_documents = []
    
    for doc in knowledge_base:
        score = 0
        context_lower = doc['context'].lower()
        product_data = doc['product_data']
        
        # Boost analytical summary for analytical queries
        if is_analytical and doc['id'] == 'ANALYTICAL_SUMMARY':
            score += 200
        
        # Exact matches
        if doc['sku'].lower() in query_lower:
            score += 100
        if doc['name'].lower() in query_lower:
            score += 50
        if doc['brand'].lower() in query_lower:
            score += 30
        
        # Word-level matches
        name_words = set(doc['name'].lower().split())
        matching_words = query_words.intersection(name_words)
        score += len(matching_words) * 10
        
        # Marketplace names
        for alias in product_data.get('marketplace_names', []):
            if alias.lower() in query_lower:
                score += 20
        
        # Context keyword match
        context_words = set(context_lower.split())
        context_matches = query_words.intersection(context_words)
        score += len(context_matches) * 2
        
        # Specific query type boosts
        if 'complaint' in query_lower or 'problem' in query_lower or 'issue' in query_lower:
            if product_data.get('type') == 'analytics':
                score += 50
            elif product_data.get('negative_count', 0) > 0:
                score += product_data.get('negative_count', 0) * 5
        
        if 'best' in query_lower or 'recommend' in query_lower:
            if product_data.get('type') == 'analytics':
                score += 50
            elif product_data.get('avg_rating', 0) >= 4:
                score += int(product_data.get('avg_rating', 0) * 10)
        
        if score > 0:
            scored_documents.append({
                'doc': doc,
                'score': score,
                'name': doc['name'],
                'brand': doc['brand']
            })
    
    # Sort by score
    scored_documents.sort(key=lambda x: x['score'], reverse=True)
    
    if not scored_documents:
        return f"No relevant information found for: '{query}'"
    
    # Get top K results
    top_results = scored_documents[:top_k]
    
    # Format context
    context_chunks = []
    for i, result in enumerate(top_results, 1):
        context_chunks.append(
            f"[Document {i}] {result['doc']['context']} "
            f"(Relevance Score: {result['score']})"
        )
    
    formatted_context = "\n\n---\n\n".join(context_chunks)
    
    # Add header
    if is_analytical:
        header = "ANALYTICAL QUERY DETECTED - Comprehensive insights included.\n\n"
    else:
        found_products = ", ".join([f"'{r['name']}'" for r in top_results[:3] if r['name'] != 'Database-Wide Analytics'])
        header = f"Found relevant information. Top matches: {found_products}\n\n"
    
    return f"{header}Product Knowledge Context:\n\n{formatted_context}"

def generate_response(query: str, context: str):
    """Calls AI API with enhanced analytical prompt."""
    system_prompt = (
        "You are an advanced RAG (Retrieval-Augmented Generation) Chatbot specializing in product analytics and customer insights. "
        "Your goal is to provide data-driven answers using ONLY the provided context. "
        "The context includes individual product details, overall statistics, complaint analysis, sentiment data, and emotion patterns. "
        "When asked analytical questions like 'what are the most common complaints' or 'which product is best', "
        "use the ANALYTICAL SUMMARY section and compare products objectively. "
        "Always cite specific numbers, ratings, and sentiment scores. "
        "If comparing products, present data in a structured, easy-to-read format. "
        "If the context lacks information, state this clearly. "
        "Be professional, data-focused, and actionable in your responses."
    )
    
    user_query = f"{context}\n\nUser Question: {query}"
    
    # Throttling
    if 'last_api_call' in st.session_state:
        elapsed = time.time() - st.session_state.last_api_call
        if elapsed < 2:
            time.sleep(2 - elapsed)
    
    st.session_state.last_api_call = time.time()
    
    # Try Gemini models
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
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not parse response.')
                return text
            elif response.status_code == 429:
                print(f"Rate limit on {model_name}")
                time.sleep(3)
                continue
            else:
                print(f"API Error on {model_name}: {response.status_code}")
                continue
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Try Claude fallback
    try:
        claude_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if claude_key:
            print("Trying Claude API...")
            claude_url = "https://api.anthropic.com/v1/messages"
            
            claude_payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1524,
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
        print(f"Claude API failed: {e}")
    
    return "⚠️ **API Rate Limit Exceeded**\n\nPlease wait 60 seconds and try again."

# --- Streamlit UI ---

if '__app_id' not in globals():
    __app_id = 'default-app-id'
if '__firebase_config' not in globals():
    __firebase_config = '{}'

if 'user_id' not in st.session_state:
    st.session_state.user_id = "local-dev-user"
    st.session_state.is_auth_ready = True

st.set_page_config(page_title="Product Analytics RAG Chatbot", page_icon="📊", layout="wide")
st.title("📊 Advanced Product Analytics RAG Chatbot")
st.caption("AI-powered product analysis with sentiment, emotion, and complaint insights")

# Initialize knowledge base
if 'knowledge_base' not in st.session_state or st.sidebar.button("🔄 Refresh Database"):
    with st.spinner("Loading data and building analytics..."):
        product_data = fetch_product_data_from_mysql()
        insights = fetch_analytical_insights()
        
        st.session_state.embeddings_model = True
        st.session_state.knowledge_base = create_product_knowledge_base(
            product_data, insights, st.session_state.embeddings_model
        )
        st.session_state.product_data = product_data
        st.session_state.insights = insights

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I can analyze product reviews, complaints, sentiments, and provide data-driven insights. Try asking analytical questions!"}
    ]

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input
if prompt := st.chat_input("Ask about products, complaints, or analytics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing data..."):
            context = retrieve_relevant_context(prompt, st.session_state.knowledge_base, top_k=3)
            
            with st.expander("🔍 Retrieved Context"):
                st.code(context, language='text')

            with st.spinner("🤖 Generating insights..."):
                response = generate_response(prompt, context)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# Enhanced Sidebar
st.sidebar.header("📊 System Status")

if st.session_state.get('insights'):
    insights = st.session_state.insights
    if 'overall' in insights:
        st.sidebar.metric("Total Products", insights['overall'].get('total_products', 0))
        st.sidebar.metric("Total Reviews", insights['overall'].get('total_reviews', 0))
        st.sidebar.metric("Avg Rating", f"{insights['overall'].get('avg_rating', 0):.2f}/5.0")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Try these analytical queries:**")
st.sidebar.markdown("- What are the most common complaints?")
st.sidebar.markdown("- Which backpack has the best rating?")
st.sidebar.markdown("- Analyze customer emotions across products")
st.sidebar.markdown("- Compare [product A] vs [product B]")
st.sidebar.markdown("- What issues do customers face with [product]?")
st.sidebar.markdown("- Show me the worst rated products")
st.sidebar.markdown("- What percentage of reviews are positive?")
