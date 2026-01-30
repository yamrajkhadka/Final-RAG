# Final-RAG: Nepal Legal AI System

**Full-Stack Retrieval-Augmented Generation (RAG) system for Nepal-specific legal queries.**  
This project integrates **document retrieval, embeddings, LLMs, API deployment, and a mobile frontend** to provide ChatGPT-like interaction for legal documents in Nepal.

---

## 🚀 Project Overview

**Final-RAG** is a complete AI solution that allows users to query Nepal-specific legal documents using natural language. It leverages a **RAG pipeline** to retrieve relevant content from PDFs, uses an **LLM** to generate human-like answers, and exposes functionality via a **REST API**. Users can access the system through a **web interface** or a **mobile app**.

**Key Features:**
- Upload PDFs and automatically extract legal text.
- Generate embeddings and chunk documents for retrieval.
- Use LLM (LLaMA 3 / Hugging Face) to answer queries.
- Expose API for mobile/web integration.
- React Native mobile frontend mimicking ChatGPT with chat history and authentication.

---
## 💻 Demo & API Links
| Platform                  | Link                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Streamlit Web UI**      | [Click Here](https://20final-rag-jxu8qwyinezorrjfhssxjg.streamlit.app)                                               |
| **Hugging Face API**      | [Click Here](https://huggingface.co/spaces/yamraj047/api-problem-fix)                                                 |
| **Expo Mobile App Build** | [Click Here](https://expo.dev/accounts/yamraj047/projects/lexnepal-ai/builds/0bed6947-3db8-40d2-8b71-b206e399f597) |

___
## ⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/yamrajkhadka/Final-RAG.git
cd Final-RAG
2. Install Dependencies
pip install -r requirements.txt
3. Run Streamlit Web UI (Local)
streamlit run 1-rag-system/app.py
4. API Usage Example
cURL
curl -X POST https://huggingface.co/spaces/yamraj047/lexnepal-api/query \
-H "Content-Type: application/json" \
-d '{"query": "What is the penalty for theft in Nepal?"}'
JavaScript / React Native
const response = await fetch("https://huggingface.co/spaces/yamraj047/lexnepal-api/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What is the penalty for theft in Nepal?" })
});
const data = await response.json();
console.log(data.answer);

## 📱 Mobile App Features
ChatGPT-style chat interface.
Sign up / Sign in / Sign out.
New chat and chat history management.
Fetch answers from Hugging Face API.
Ready for Android/iOS via Expo.
🛠 Technologies Used
Backend / RAG: Python, FAISS, SentenceTransformers, LLaMA 3
Frontend Web: Streamlit
Frontend Mobile: React Native / Expo
Deployment: Streamlit Cloud, Hugging Face Spaces, Expo
Others: JSON, PDF parsing libraries
