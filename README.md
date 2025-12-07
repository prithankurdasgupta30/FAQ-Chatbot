# 📚 Local AI-Powered FAQ Assistant (Streamlit + Semantic Search)

A **fully offline, AI-powered FAQ chatbot** built using **SentenceTransformers, semantic search, and Streamlit**. This application allows users to ask questions in natural language and instantly retrieves the most relevant answer from a CSV-based FAQ dataset — **without using any external APIs, cloud services, or paid LLMs**.

This project uses **vector embeddings + cosine similarity** to perform intelligent semantic matching between user queries and stored FAQs.

---

## 🚀 Features

- ✅ Fully **offline AI system** (No API keys required)
- ✅ **Semantic search** using vector embeddings
- ✅ **Real-time web UI** built with Streamlit
- ✅ **Confidence thresholding** for reliable answers
- ✅ **Debug mode** to inspect retrieved FAQ matches
- ✅ Works with **any CSV-based FAQ dataset**
- ✅ Lightweight, fast, and cost-free deployment

---

## 🧠 How It Works

1. Loads FAQ data from a CSV file.
2. Converts each FAQ entry into vector embeddings using SentenceTransformers.
3. Converts the user’s question into an embedding at runtime.
4. Uses **cosine similarity** to find the closest FAQ match.
5. Displays the most relevant answer instantly in the Streamlit UI.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit**
- **SentenceTransformers (all-MiniLM-L6-v2)**
- **Scikit-learn**
- **Pandas**
- **NumPy**

---

## ✅ System Requirements

- Python **3.9 or higher**
- Windows / macOS / Linux
- Minimum **4 GB RAM** recommended
- Internet required **only once** to download the embedding model

---

