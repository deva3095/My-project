# streamlit_app.py
import os
import time
import textwrap
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from openai import OpenAI

# RAG imports
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# Load secrets (from .env or environment)
# -------------------------
load_dotenv()  # reads .env into env vars (if present)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # MUST set this
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment. Set it or use a .env file.")

# Create OpenAI client (new SDK)
client = OpenAI(api_key=OPENAI_KEY)

# -------------------------
# Constants & RAG config
# -------------------------
CONTEXT_TABLE = "REVIEWS_WITH_SENTIMENT"  # same table you're reading
COMMON_TEXT_COLS = ["review_text", "review", "text", "comment", "feedback", "body", "content", "REVIEW"]
DEFAULT_RAG_K = 3
DEFAULT_HISTORY_LENGTH = 5

# -------------------------
# Snowflake helper
# -------------------------
@st.cache_data(ttl=300)
def fetch_reviews_from_snowflake(limit=None) -> pd.DataFrame:
    """Fetch reviews table into a pandas DataFrame using snowflake-connector."""
    sf = st.secrets["snowflake"]

    conn = snowflake.connector.connect(
        user=sf["user"],
        password=sf["password"],
        account=sf["account"],
        warehouse=sf["warehouse"],
        database=sf["database"],
        schema=sf["schema"],
    )
    cur = conn.cursor()
    try:
        sql = f"SELECT * FROM {CONTEXT_TABLE}"
        if limit:
            sql = f"{sql} LIMIT {int(limit)}"
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        return df
    finally:
        cur.close()
        conn.close()

# -------------------------
# TF-IDF vector store (cached)
# -------------------------
@st.cache_resource
def build_vectorizer_and_matrix(docs: List[str]):
    """Build TF-IDF vectorizer + matrix for a list of docs."""
    if not docs:
        return None, None
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix

def _get_corpus_from_df(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """Return a list of document strings and their original indices.
       Heuristic: prefer common text columns; else concatenate all string columns."""
    if df.empty:
        return [], []

    # Look for common column names
    for col in COMMON_TEXT_COLS:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            docs = df[col].fillna("").astype(str).tolist()
            return docs, df.index.tolist()

    # Fallback: combine all string-like columns
    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if string_cols:
        docs = df[string_cols].fillna("").agg(" ".join, axis=1).astype(str).tolist()
        return docs, df.index.tolist()

    # Last resort: convert all columns to string
    docs = df.fillna("").astype(str).agg(" ".join, axis=1).tolist()
    return docs, df.index.tolist()

def retrieve_top_k_contexts(df: pd.DataFrame, query: str, top_k: int = DEFAULT_RAG_K) -> pd.DataFrame:
    """Retrieve top_k most similar rows (by TF-IDF) from df."""
    if df.empty or not query:
        return pd.DataFrame()

    docs, indices = _get_corpus_from_df(df)
    if not docs:
        return pd.DataFrame()

    vectorizer, matrix = build_vectorizer_and_matrix(docs)
    if vectorizer is None or matrix is None:
        return pd.DataFrame()

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()  # shape (n_docs,)
    if sims.size == 0:
        return pd.DataFrame()

    top_idx = np.argsort(-sims)[:top_k]
    doc_indices = [indices[i] for i in top_idx]
    results = df.loc[doc_indices].copy()
    results["_rag_score"] = sims[top_idx]
    return results

# -------------------------
# OpenAI helper with retry/backoff
# -------------------------
def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=600, retries=3):
    last_exc = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            # The SDK returns message content at resp.choices[0].message.content in your example
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            backoff = 2 ** i
            time.sleep(backoff)
    raise last_exc

# -------------------------
# Session state (chat history & settings)
# -------------------------
def initialize_session_state():
    if "messages" not in st.session_state:
        # messages is a list of dicts: {"role": "user"|"assistant", "content": "..."}
        st.session_state.messages = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o-mini"
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "rag_top_k" not in st.session_state:
        st.session_state.rag_top_k = DEFAULT_RAG_K
    if "num_chat_messages" not in st.session_state:
        st.session_state.num_chat_messages = DEFAULT_HISTORY_LENGTH

def get_formatted_chat_history() -> str:
    """Return last N messages formatted for the LLM (role: content)."""
    if not st.session_state.get("messages"):
        return ""
    n = st.session_state.get("num_chat_messages", DEFAULT_HISTORY_LENGTH)
    msgs = st.session_state.messages[-n:]
    formatted = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
    return formatted

# -------------------------
# Prompt builder mixing chat history + RAG contexts
# -------------------------
def create_prompt(user_question: str, sample_context: str, chat_history: str, retrieved_docs: pd.DataFrame) -> str:
    """Creates the prompt combining system inst, chat history, retrieved docs, and small sample context."""
    rag_section = ""
    if not retrieved_docs.empty:
        rows = []
        # pick a text column to show snippets from
        text_col = None
        for col in COMMON_TEXT_COLS:
            if col in retrieved_docs.columns:
                text_col = col
                break
        if text_col is None:
            string_cols = [c for c in retrieved_docs.columns if pd.api.types.is_string_dtype(retrieved_docs[c])]
            text_col = string_cols[0] if string_cols else None

        for _, r in retrieved_docs.iterrows():
            snippet = (str(r[text_col])[:500] + "...") if text_col else r.to_json()
            rows.append(f"- score: {r.get('_rag_score', 0):.4f} | {snippet}")
        rag_section = "\n\n<retrieved_docs>\n" + "\n".join(rows) + "\n</retrieved_docs>\n"

    prompt_template = f"""
You are a helpful data analysis assistant specialized in interpreting customer reviews and sentiment.
Use the chat history (if present) and the retrieved review snippets to answer the user's question as precisely as possible.
If a retrieved snippet supports your answer, reference it briefly.

<chat_history>
{chat_history}
</chat_history>

{rag_section}

<context_sample>
{sample_context}
</context_sample>

Question:
{user_question}

Answer concisely:
"""
    prompt = prompt_template.strip()
    if st.session_state.get("debug"):
        st.sidebar.text_area("DEBUG: prompt", prompt, height=400)
    return prompt

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Avalanche - Reviews (RAG Chat)", layout="wide")
st.title("Avalanche Streamlit App — Chat + RAG")

initialize_session_state()

# Sidebar controls
st.sidebar.header("Controls")
st.sidebar.checkbox("Debug mode", key="debug")
st.sidebar.checkbox("Enable RAG retrieval", key="use_rag")
st.sidebar.number_input("RAG: number of docs to retrieve", key="rag_top_k", min_value=1, max_value=10, value=DEFAULT_RAG_K, step=1)
st.sidebar.number_input("Max chat messages to include in prompt", key="num_chat_messages", min_value=1, max_value=25, value=DEFAULT_HISTORY_LENGTH, step=1)
st.sidebar.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-preview"], key="model_name")  # adjust as you like
st.sidebar.button("Clear conversation", on_click=lambda: st.session_state.update({"messages": []}))

# Load data (cached)
with st.spinner("Loading data from Snowflake..."):
    try:
        df_reviews = fetch_reviews_from_snowflake(limit=2000)
    except Exception as e:
        st.error("Failed to load data from Snowflake. Check credentials and network.")
        st.exception(e)
        st.stop()

# Basic cleaning
if "REVIEW_DATE" in df_reviews.columns:
    df_reviews["REVIEW_DATE"] = pd.to_datetime(df_reviews["REVIEW_DATE"], errors="coerce")
if "SHIPPING_DATE" in df_reviews.columns:
    df_reviews["SHIPPING_DATE"] = pd.to_datetime(df_reviews["SHIPPING_DATE"], errors="coerce")

# Build TF-IDF matrix in background (cached) if data exists and RAG enabled
if not df_reviews.empty and st.session_state.use_rag:
    docs, _ = _get_corpus_from_df(df_reviews)
    if docs:
        # this call caches vectorizer and matrix
        build_vectorizer_and_matrix(docs)

# Sidebar filters
st.sidebar.header("Filters")
product_list = ["All Products"] + sorted(df_reviews["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in df_reviews.columns else ["All Products"]
selected_product = st.sidebar.selectbox("Product", product_list)

# Main visuals
st.subheader("Average Sentiment by Product")
if "PRODUCT" in df_reviews.columns and "SENTIMENT_SCORE" in df_reviews.columns:
    product_sentiment = df_reviews.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(product_sentiment)*0.25)))
    product_sentiment.plot(kind="barh", ax=ax)
    ax.set_xlabel("Sentiment Score")
    st.pyplot(fig)
else:
    st.info("PRODUCT or SENTIMENT_SCORE column missing in the table.")

# Filtered table
if selected_product != "All Products":
    filtered_data = df_reviews[df_reviews["PRODUCT"] == selected_product]
else:
    filtered_data = df_reviews

st.subheader(f"Reviews ({len(filtered_data)}) — {selected_product}")
st.dataframe(filtered_data.head(200))

# Distribution
st.subheader("Sentiment Distribution")
if "SENTIMENT_SCORE" in filtered_data.columns and not filtered_data["SENTIMENT_SCORE"].dropna().empty:
    fig2, ax2 = plt.subplots()
    filtered_data["SENTIMENT_SCORE"].hist(ax=ax2, bins=20)
    ax2.set_xlabel("Sentiment Score")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
else:
    st.info("No SENTIMENT_SCORE data available for histogram.")

# Prepare a small sample string for LLM (limit size)
sample_string = ""
if not df_reviews.empty:
    sample_df = df_reviews.sample(n=min(100, len(df_reviews)), random_state=1)
    cols_to_show = sample_df.columns[:6]  # first 6 columns for brevity
    sample_string = sample_df[cols_to_show].to_string(index=False)
    sample_string = textwrap.shorten(sample_string, width=3000, placeholder=" ... [truncated]")

# -------------------------
# Chat + Q&A UI
# -------------------------
st.subheader("Chat with your data (History + RAG)")
icons = {"assistant": "🤖", "user": "👤"}

# Display chat history
for message in st.session_state.messages:
    avatar = icons.get(message["role"], "")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input area
user_input = st.chat_input("Ask about reviews, sentiment, or the dataset...")

if user_input:
    # Append user message to history (so next LLM turn sees it)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar=icons["user"]):
        st.markdown(user_input)

    # Retrieve RAG docs if enabled
    retrieved = pd.DataFrame()
    if st.session_state.use_rag and not df_reviews.empty:
        try:
            retrieved = retrieve_top_k_contexts(df_reviews, user_input, top_k=st.session_state.rag_top_k)
        except Exception as e:
            if st.session_state.debug:
                st.sidebar.error(f"RAG retrieval error: {e}")
            retrieved = pd.DataFrame()

    # Compose prompt containing chat history + retrieved snippets + small context sample
    chat_history_str = get_formatted_chat_history()
    # Prefer retrieved snippets for context, else fallback to sample_string
    sample_context_for_prompt = ""
    if not retrieved.empty:
        # compact representation
        rows = []
        text_col = None
        for col in COMMON_TEXT_COLS:
            if col in retrieved.columns:
                text_col = col
                break
        if text_col is None:
            string_cols = [c for c in retrieved.columns if pd.api.types.is_string_dtype(retrieved[c])]
            text_col = string_cols[0] if string_cols else None
        for _, r in retrieved.iterrows():
            snippet = (str(r[text_col])[:800] + "...") if text_col else r.to_json()
            rows.append(f"score: {r.get('_rag_score', 0):.4f} | {snippet}")
        sample_context_for_prompt = "\n".join(rows)
    else:
        sample_context_for_prompt = sample_string

    prompt = create_prompt(
        user_question=user_input,
        sample_context=sample_context_for_prompt,
        chat_history=chat_history_str,
        retrieved_docs=retrieved
    )

    # Call OpenAI
    with st.chat_message("assistant", avatar=icons["assistant"]):
        placeholder = st.empty()
        with st.spinner("Contacting OpenAI..."):
            try:
                model = st.session_state.model_name
                answer = call_openai_chat(prompt, model=model, max_tokens=800)
                placeholder.markdown(answer)
                # Save assistant response into history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("OpenAI Error: could not get a response.")
                st.exception(e)

st.markdown(
    """
    **Notes**
    finally it works
    """
)
