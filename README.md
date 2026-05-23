# Hybrid NLP Recommendation System

A hybrid NLP-powered recommendation system for e-commerce product reviews using machine learning and semantic similarity techniques.

This project combines TF-IDF, cosine similarity, BERT embeddings, sentiment analysis, and Streamlit to recommend similar products based on customer review text.

---

## Features

- NLP-based product recommendation
- TF-IDF vectorization
- Cosine similarity matching
- BERT semantic embeddings
- Sentiment analysis on customer reviews
- Streamlit interactive dashboard
- Review preprocessing pipeline
- Hybrid recommendation architecture

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers (BERT)
- PyTorch
- Streamlit
- Matplotlib

---

## Dataset

This project uses an Amazon product reviews dataset for experimentation and recommendation modeling.

Dataset includes:

- Product names
- Product categories
- Customer reviews
- Ratings
- Metadata

---

## Project Structure

```text
hybrid-nlp-recommendation-system/
│
├── data/
│   └── raw/
│       └── 7817_1.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── cosine_similarity_matrix.npy
│   └── processed_product_reviews.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How It Works

### 1. Data Preprocessing

Customer reviews are cleaned and prepared for NLP processing.

### 2. TF-IDF Vectorization

Reviews are transformed into numerical vectors using TF-IDF.

### 3. Cosine Similarity

Similarity scores are calculated between products based on review text.

### 4. BERT Embeddings

Semantic embeddings improve understanding of contextual meaning between reviews.

### 5. Hybrid Recommendation System

TF-IDF and BERT similarity approaches are combined to generate recommendations.

### 6. Sentiment Analysis

Customer review sentiment is analyzed to identify positive and negative opinions.

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit app

```bash
streamlit run app.py
```

---

## Limitations

- Sentiment analysis is based on a pre-trained model and may misclassify some reviews.
- Some positive-sounding reviews may receive incorrect sentiment predictions due to contextual limitations.
- Recommendations are based only on review text similarity.
- The project does not use real behavioural analytics such as clicks, purchases, sessions, or user interaction history.
- The recommendation system is not personalised to individual users.
- Visualisations are exploratory and may require refinement for production-level applications.

---

## Future Improvements

- Collaborative filtering
- User-personalized recommendations
- Real-time recommendation API
- Cloud deployment
- Behavioural analytics integration
- Advanced evaluation metrics

---

## Author

Olamilekan Alaga

MSc Data Science graduate with interests in NLP, recommendation systems, behavioural analytics, and machine learning.