# 🤖 AI-Based Smart Recommendation System

An end-to-end E-commerce and Business Intelligence recommendation system built using transactional retail data. This project implements dual-algorithm recommendation modeling and features an interactive web dashboard built with **Streamlit** hosted via Google Colab.

## 🚀 Features
- **Data Exploration:** Interactive preview of the real-world Kaggle online retail transactional logs.
- **Dual-Engine Architecture:**
  - **Popularity-Based Engine:** Identifies globally trending hot items based on transaction frequencies and total items sold.
  - **AI Matrix Factorization (SVD):** Leverages Singular Value Decomposition to calculate user-behavior vectors and predict personalized complementary products with dynamic confidence scores.
- **Interactive Select Interface:** Completely click-driven product search filter allowing dynamic, real-time prediction updates without manual text inputs.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-Learn (TruncatedSVD), Matplotlib, Seaborn
- **Framework:** Streamlit Dashboard Engine
- **Environment:** Google Colab Cloud Ecosystem

## 📂 Repository Structure
├── AI_Smart_Recommendation_System.ipynb # Main Google Colab Notebook
├── app.py                              # Streamlit Dashboard UI Engine
└── README.md                           # Project Documentation
