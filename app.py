
import streamlit as st
import joblib
from scipy import sparse
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_models():
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
    tfidf_mat = sparse.load_npz('models/tfidf_matrix.npz')
    products = pd.read_parquet('models/product_lookup.parquet')
    return tfidf, tfidf_mat, products

tfidf, tfidf_mat, products = load_models()
st.title('Gift Recommender')
query = st.text_input('Describe recipient + occasion + budget', 'birthday gift for a coffee lover')
k = st.slider('Number of results', 3, 20, 8)
if st.button('Recommend'):
    q_vec = tfidf.transform([query])
    sims = linear_kernel(q_vec, tfidf_mat).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    res = products.iloc[top_idx].copy()
    res['score'] = sims[top_idx]
    for _, row in res.head(k).iterrows():
        st.write(f"**{row.get('title','')}** — {row.get('price','')}, score: {row['score']:.3f}")
