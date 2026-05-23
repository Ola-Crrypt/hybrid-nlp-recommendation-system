import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from textblob import TextBlob


st.set_page_config(
    page_title="Hybrid Product Recommender",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/7817_1.csv")

    df = df[["name", "reviews.text", "reviews.rating"]].copy()

    df["name"] = df["name"].fillna("")
    df["reviews.text"] = df["reviews.text"].fillna("")
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")

    df = df.dropna(subset=["reviews.rating"])

    def get_sentiment(text):
        score = TextBlob(str(text)).sentiment.polarity

        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment"] = df["reviews.text"].apply(get_sentiment)

    product_df = (
        df.groupby("name")
        .agg(
            product_text=("reviews.text", lambda x: " ".join(x.astype(str))),
            avg_rating=("reviews.rating", "mean"),
            review_count=("reviews.rating", "count"),
            positive_reviews=("sentiment", lambda x: (x == "Positive").sum()),
            neutral_reviews=("sentiment", lambda x: (x == "Neutral").sum()),
            negative_reviews=("sentiment", lambda x: (x == "Negative").sum())
        )
        .reset_index()
    )

    product_df = product_df[product_df["name"] != ""].reset_index(drop=True)

    product_df["positive_rate"] = (
        product_df["positive_reviews"] / product_df["review_count"]
    )

    product_df["negative_rate"] = (
        product_df["negative_reviews"] / product_df["review_count"]
    )

    return df, product_df


@st.cache_resource
def load_bert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def build_tfidf_similarity(product_df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

    tfidf_matrix = tfidf.fit_transform(
        product_df["name"] + " " + product_df["product_text"]
    )

    return cosine_similarity(tfidf_matrix)


@st.cache_data
def build_bert_similarity(product_texts):
    model = load_bert_model()

    embeddings = model.encode(
        product_texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return cosine_similarity(embeddings)


def recommend_products(product_name, product_df, similarity_matrix, top_n=5):
    matches = product_df[
        product_df["name"].str.lower() == product_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    product_index = matches.index[0]

    similarity_scores = list(enumerate(similarity_matrix[product_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n + 1]

    recommended_indices = [i[0] for i in similarity_scores]
    scores = [i[1] for i in similarity_scores]

    recommendations = product_df.iloc[recommended_indices][
        [
            "name",
            "avg_rating",
            "review_count",
            "positive_rate",
            "negative_rate"
        ]
    ].copy()

    recommendations["recommendation_score"] = scores

    return recommendations


df, product_df = load_data()

tfidf_similarity = build_tfidf_similarity(product_df)

bert_texts = (
    product_df["name"] + " " + product_df["product_text"]
).tolist()

bert_similarity = build_bert_similarity(bert_texts)

hybrid_similarity = (0.5 * tfidf_similarity) + (0.5 * bert_similarity)


tab1, tab2 = st.tabs(["🔍 Recommender", "📊 Dashboard"])


with tab1:
    st.title("Hybrid Product Recommendation System")

    model_choice = st.selectbox(
        "Choose recommendation model",
        ["Hybrid TF-IDF + BERT", "TF-IDF Only", "BERT Only"]
    )

    product_names = product_df["name"].sort_values().unique()

    selected_product = st.selectbox(
        "Select a product",
        product_names
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if model_choice == "TF-IDF Only":
        selected_similarity = tfidf_similarity
    elif model_choice == "BERT Only":
        selected_similarity = bert_similarity
    else:
        selected_similarity = hybrid_similarity

    if st.button("Recommend Products"):
        recommendations = recommend_products(
            selected_product,
            product_df,
            selected_similarity,
            top_n
        )

        st.subheader("Recommended Products")

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.dataframe(recommendations)


with tab2:
    st.title("Product Review Dashboard")

    total_products = product_df["name"].nunique()
    total_reviews = len(df)
    avg_rating = df["reviews.rating"].mean()

    positive_pct = (df["sentiment"] == "Positive").mean() * 100
    negative_pct = (df["sentiment"] == "Negative").mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", total_products)
    col2.metric("Total Reviews", total_reviews)
    col3.metric("Average Rating", round(avg_rating, 2))

    col4, col5 = st.columns(2)

    col4.metric("Positive Reviews %", f"{positive_pct:.2f}%")
    col5.metric("Negative Reviews %", f"{negative_pct:.2f}%")

    st.subheader("Rating Distribution")

    rating_counts = df["reviews.rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

    st.subheader("Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Top Products by Review Count")

    top_reviewed = product_df.sort_values(
        by="review_count",
        ascending=False
    ).head(10)

    st.bar_chart(
        top_reviewed.set_index("name")["review_count"]
    )

    st.subheader("Top Products by Average Rating")

    top_rated = product_df[
        product_df["review_count"] >= 3
    ].sort_values(
        by="avg_rating",
        ascending=False
    ).head(10)

    st.dataframe(
        top_rated[
            [
                "name",
                "avg_rating",
                "review_count",
                "positive_rate",
                "negative_rate"
            ]
        ]
    )

    st.subheader("Most Positive Products")

    most_positive = product_df[
        product_df["review_count"] >= 5
    ].sort_values(
        by="positive_rate",
        ascending=False
    ).head(10)

    st.dataframe(
        most_positive[
            [
                "name",
                "review_count",
                "avg_rating",
                "positive_rate",
                "negative_rate"
            ]
        ]
    )

    st.subheader("Most Complained-About Products")

    most_negative = product_df[
        product_df["review_count"] >= 5
    ].sort_values(
        by="negative_rate",
        ascending=False
    ).head(10)

    st.dataframe(
        most_negative[
            [
                "name",
                "review_count",
                "avg_rating",
                "positive_rate",
                "negative_rate"
            ]
        ]
    )

    st.subheader("Product Intelligence Explorer")

    selected_dashboard_product = st.selectbox(
        "Select Product for Analysis",
        product_df["name"].sort_values().unique()
    )

    product_info = product_df[
        product_df["name"] == selected_dashboard_product
    ]

    st.dataframe(
        product_info[
            [
                "name",
                "avg_rating",
                "review_count",
                "positive_rate",
                "negative_rate"
            ]
        ]
    )

    st.subheader("Sample Customer Reviews")

    sentiment_filter = st.selectbox(
        "Filter reviews by sentiment",
        ["All", "Positive", "Neutral", "Negative"]
    )

    product_reviews = df[df["name"] == selected_dashboard_product]

    if sentiment_filter != "All":
        product_reviews = product_reviews[
            product_reviews["sentiment"] == sentiment_filter
        ]

    sample_reviews = product_reviews[
        ["reviews.text", "reviews.rating", "sentiment"]
    ].head(10)

    st.dataframe(sample_reviews)

    st.subheader("Customer Pain-Point Signal")

    negative_reviews = df[
        (df["name"] == selected_dashboard_product) &
        (df["sentiment"] == "Negative")
    ]

    st.metric("Negative Review Count", len(negative_reviews))

    if len(negative_reviews) > 0:
        st.write("Recent negative review examples:")
        st.dataframe(
            negative_reviews[
                ["reviews.text", "reviews.rating", "sentiment"]
            ].head(5)
        )
    else:
        st.success("No negative reviews detected for this product.")