import streamlit as st
st.set_page_config(
    page_title="Course FAQ Assistant",
    page_icon="❓",
    layout="centered",
)
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
CSV_PATH = "Dataset - Sheet2.csv"
QUESTION_COL = "Prompt"
ANSWER_COL = "Response"
def load_faq_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if QUESTION_COL not in df.columns or ANSWER_COL not in df.columns:
        raise ValueError(
            f"CSV must contain '{QUESTION_COL}' and '{ANSWER_COL}' columns. "
            f"Found columns: {list(df.columns)}"
        )
    return df
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def build_embeddings():
    df_local = load_faq_data(CSV_PATH)
    model_local = load_model()
    faq_texts = (
        "Question: " + df_local[QUESTION_COL].astype(str) +
        "\nAnswer: " + df_local[ANSWER_COL].astype(str)
    ).tolist()
    faq_embeddings_local = model_local.encode(
        faq_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return df_local, faq_embeddings_local
def retrieve_similar_faqs(query, df_local, model_local, faq_embeddings_local, top_k=4):
    query_emb = model_local.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, faq_embeddings_local)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_indices]
    results = []
    for idx, score in zip(top_indices, top_scores):
        row = df_local.iloc[idx]
        results.append(
            {
                "index": int(idx),
                "score": float(score),
                "prompt": row[QUESTION_COL],
                "response": row[ANSWER_COL],
            }
        )
    return results
def answer_question(question, df_local, model_local, faq_embeddings_local,
                    top_k=4, debug=False, threshold=0.4):
    faqs = retrieve_similar_faqs(
        question, df_local, model_local, faq_embeddings_local, top_k
    )
    if debug:
        st.write("### Retrieved FAQ Candidates")
        for f in faqs:
            st.write(f"**Score:** {f['score']:.3f}")
            st.write(f"**FAQ Question:** {f['prompt']}")
            st.write(f"**FAQ Answer:** {f['response']}")
            st.write("---")
    best = faqs[0]
    if best["score"] < threshold:
        return "I'm not sure based on the available FAQs."
    return best["response"]
st.title("📚 Course FAQ Assistant (Local AI)")
st.write(
    "Ask any question related to the course and I'll try to match it "
    "with the most relevant FAQ from the dataset."
)
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of FAQs to search (k)", 1, 10, 4)
threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.4, 0.05
)
debug = st.sidebar.checkbox("Show retrieved FAQ candidates (debug)", False)
try:
    df, faq_embeddings = build_embeddings()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()
user_question = st.text_input(
    "Enter your question:",
    placeholder="e.g., Do you provide job assistance and job guarantee?",
)
if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(
                user_question,
                df,
                model,
                faq_embeddings,
                top_k=top_k,
                debug=debug,
                threshold=threshold,
            )
        st.subheader("✅ Answer")
        st.write(answer)
        faqs = retrieve_similar_faqs(
            user_question, df, model, faq_embeddings, top_k=1
        )
        best = faqs[0]
        st.caption(
            f"Best match FAQ (score={best['score']:.3f}): **{best['prompt']}**"
        )
