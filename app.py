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

# -------------------------
# Load secrets (from .env or environment)
# -------------------------
load_dotenv()  # reads .env into env vars (if present)



OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # MUST set this (rotate the leaked one)

if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment. Set it or use a .env file.")

# Create OpenAI client (new SDK)
client = OpenAI(api_key=OPENAI_KEY)

# -------------------------
# Snowflake helper
# -------------------------
@st.cache_data(ttl=300)
def fetch_reviews_from_snowflake(limit=None):
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
        sql = "SELECT * FROM REVIEWS_WITH_SENTIMENT"
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
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            backoff = 2 ** i
            time.sleep(backoff)
    raise last_exc

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Avalanche - Reviews", layout="wide")
st.title("Avalanche Streamlit App")

# Load data (cached)
with st.spinner("Loading data from Snowflake..."):
    try:
        df_reviews = fetch_reviews_from_snowflake(limit=2000)  # fetch up to 2000 rows; tune as needed
    except Exception as e:
        st.error("Failed to load data from Snowflake. Check credentials and network.")
        st.exception(e)
        st.stop()

# Basic cleaning
if "REVIEW_DATE" in df_reviews.columns:
    df_reviews["REVIEW_DATE"] = pd.to_datetime(df_reviews["REVIEW_DATE"], errors="coerce")
if "SHIPPING_DATE" in df_reviews.columns:
    df_reviews["SHIPPING_DATE"] = pd.to_datetime(df_reviews["SHIPPING_DATE"], errors="coerce")

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
    cols_to_show = sample_df.columns[:6]  # first 6 columns
    sample_string = sample_df[cols_to_show].to_string(index=False)
    sample_string = textwrap.shorten(sample_string, width=3000, placeholder=" ... [truncated]")

# Q&A with OpenAI
st.subheader("Ask Questions About Your Data (via OpenAI)")
user_question = st.text_input("Enter your question:")

if user_question:
    prompt = "You are a helpful data analysis assistant.\n"
    if sample_string:
        prompt += f"Here is a sample of the dataset (up to 100 rows):\n{sample_string}\n\n"
    prompt += f"Answer this question clearly and concisely: {user_question}"

    with st.spinner("Contacting OpenAI..."):
        try:
            answer = call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=600)
            st.markdown("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error("OpenAI Error: could not get a response.")
            st.exception(e)

st.markdown(
    """
    **Notes**
    - Do not hardcode keys in source. Use environment variables or a .env file (and never commit it).
    - If running locally, ensure you set OPENAI_API_KEY and Snowflake credentials in env/.env.
    - For bulk writes back to Snowflake, use staged CSV + COPY INTO for better performance.
    """
)
