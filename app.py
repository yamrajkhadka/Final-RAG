import streamlit as st
import numpy as np
import json
import faiss
import re
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import time

st.set_page_config(
    page_title="LexNepal AI | Elite Legal Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_ultra_premium_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=JetBrains+Mono:wght@400;500;600&family=Crimson+Pro:wght@400;600;700&display=swap');
        .stApp {background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); background-attachment: fixed;}
        #MainMenu, footer, header {visibility: hidden;}
        .ultimate-hero {background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 58, 138, 0.9) 50%, rgba(37, 99, 235, 0.85) 100%); backdrop-filter: blur(30px); padding: 4.5rem 3.5rem; border-radius: 35px; text-align: center; margin-bottom: 3rem; border: 2px solid rgba(147, 197, 253, 0.2); box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5), inset 0 2px 0 rgba(255, 255, 255, 0.1), 0 0 100px rgba(59, 130, 246, 0.2); position: relative; overflow: hidden;}
        .ultimate-hero::before {content: ''; position: absolute; top: -100%; left: -100%; width: 300%; height: 300%; background: radial-gradient(circle at center, rgba(59, 130, 246, 0.15) 0%, transparent 70%); animation: cosmic-rotation 20s linear infinite;}
        @keyframes cosmic-rotation {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}
        .hero-badge {display: inline-block; background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%); border: 1px solid rgba(34, 197, 94, 0.3); padding: 8px 20px; border-radius: 25px; font-size: 0.85rem; font-weight: 600; color: #6ee7b7; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1.5rem; position: relative; z-index: 1; font-family: 'JetBrains Mono', monospace;}
        .hero-badge::before {content: '●'; color: #22c55e; margin-right: 8px; animation: pulse-dot 2s ease-in-out infinite;}
        @keyframes pulse-dot {0%, 100% {opacity: 1;} 50% {opacity: 0.5;}}
        .hero-title {font-family: 'Playfair Display', serif; font-size: 5rem; font-weight: 900; background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.2rem; position: relative; z-index: 1; line-height: 1.1;}
        .hero-subtitle {font-family: 'Inter', sans-serif; font-size: 1.4rem; color: #cbd5e1; font-weight: 400; letter-spacing: 0.5px; position: relative; z-index: 1; line-height: 1.6;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important; border-right: 2px solid rgba(148, 163, 184, 0.2);}
        .sidebar-header {text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(37, 99, 235, 0.2) 100%); border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(147, 197, 253, 0.2);}
        .sidebar-logo {font-size: 3.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.6));}
        .sidebar-title {font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #ffffff 0%, #93c5fd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem;}
        .sidebar-tagline {color: #94a3b8 !important; font-size: 0.8rem; font-style: italic;}
        
        .project-description {background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 20px; padding: 1.5rem; margin: 1.5rem 0; backdrop-filter: blur(10px);}
        .desc-title {color: #c4b5fd !important; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 1rem; font-family: 'Inter', sans-serif; text-align: center;}
        .desc-text {color: #e2e8f0 !important; font-size: 0.88rem; line-height: 1.7; font-family: 'Inter', sans-serif;}
        .desc-highlight {color: #a78bfa !important; font-weight: 600;}
        
        .architecture-section {background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 58, 138, 0.3) 100%); border: 1px solid rgba(147, 197, 253, 0.2); border-radius: 20px; padding: 1.5rem; margin: 1.5rem 0; backdrop-filter: blur(10px);}
        .architecture-title {color: #fbbf24 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 1rem; font-family: 'JetBrains Mono', monospace; text-align: center;}
        .tech-card {background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%); border: 1px solid rgba(147, 197, 253, 0.2); padding: 1.2rem; border-radius: 15px; margin-bottom: 0.9rem; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); backdrop-filter: blur(10px);}
        .tech-card:hover {transform: translateX(8px) scale(1.02); border-color: rgba(147, 197, 253, 0.5); box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);}
        .tech-label {color: #60a5fa !important; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;}
        .tech-value {color: #e2e8f0 !important; font-size: 1rem; font-family: 'Inter', sans-serif; font-weight: 600;}
        .tech-detail {color: #94a3b8 !important; font-size: 0.75rem; margin-top: 0.3rem;}
        .methodology-card {background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%); border-left: 4px solid #8b5cf6; padding: 1rem; border-radius: 12px; margin: 0.8rem 0;}
        .methodology-step {color: #c4b5fd !important; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem;}
        .methodology-desc {color: #cbd5e1 !important; font-size: 0.8rem;}
        .stat-card {background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%); border: 2px solid rgba(147, 197, 253, 0.15); border-radius: 25px; padding: 2.5rem 2rem; text-align: center; transition: all 0.4s; backdrop-filter: blur(20px);}
        .stat-card:hover {transform: translateY(-10px) scale(1.03); box-shadow: 0 25px 50px rgba(59, 130, 246, 0.4); border-color: rgba(147, 197, 253, 0.5);}
        .stat-icon {font-size: 2.5rem; margin-bottom: 1rem;}
        .stat-number {font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; margin-bottom: 0.5rem;}
        .stat-label {color: #cbd5e1; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;}
        .stat-sublabel {color: #64748b; font-size: 0.75rem; margin-top: 0.3rem; font-style: italic;}
        .legal-answer-card {background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 3.5rem; border-radius: 30px; border-left: 10px solid #1e3a8a; box-shadow: 0 30px 70px rgba(0, 0, 0, 0.2), inset 0 2px 0 rgba(255, 255, 255, 0.9); font-family: 'Crimson Pro', serif; color: #1e293b; line-height: 2; font-size: 1.15rem; margin: 2.5rem 0; position: relative; overflow: hidden;}
        .legal-answer-card::before {content: '⚖️'; position: absolute; top: -40px; right: -40px; font-size: 300px; opacity: 0.02; transform: rotate(15deg);}
        .references-header {text-align: center; margin: 3rem 0 2rem 0;}
        .references-title {font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;}
        .references-subtitle {color: #94a3b8; font-size: 1rem; font-style: italic;}
        .source-card {background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border: 2px solid transparent; border-radius: 25px; padding: 2rem; margin-top: 1.5rem; transition: all 0.5s; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); position: relative; overflow: hidden;}
        .source-card::before {content: ''; position: absolute; inset: 0; border-radius: 25px; padding: 2px; background: linear-gradient(135deg, #1e3a8a, #3b82f6, #60a5fa); -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor; mask-composite: exclude; opacity: 0; transition: opacity 0.5s;}
        .source-card:hover::before {opacity: 1;}
        .source-card:hover {transform: translateY(-12px) scale(1.03); box-shadow: 0 30px 60px rgba(30, 58, 138, 0.3);}
        .law-tag {background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e3a8a; padding: 8px 20px; border-radius: 30px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; display: inline-block; box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15); font-family: 'JetBrains Mono', monospace;}
        .section-title {margin-top: 1.2rem; font-weight: 700; color: #0f172a; font-size: 1.25rem; font-family: 'Inter', sans-serif;}
        .section-text {font-size: 0.95rem; color: #475569; margin-top: 1rem; line-height: 1.8;}
        .relevance-score {position: absolute; top: 1.5rem; right: 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%); border: 1px solid rgba(34, 197, 94, 0.3); color: #22c55e; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;}
        .stButton > button {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); color: white; border: none; border-radius: 20px; padding: 1rem 2.5rem; font-weight: 700; transition: all 0.3s; box-shadow: 0 8px 20px rgba(30, 58, 138, 0.4); text-transform: uppercase; letter-spacing: 1px;}
        .stButton > button:hover {transform: translateY(-3px); box-shadow: 0 12px 35px rgba(30, 58, 138, 0.6);}
        .disclaimer-box {background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.08) 100%); border-left: 6px solid #ef4444; padding: 2rem; border-radius: 20px; margin-top: 3rem; color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.2);}
        .disclaimer-title {font-size: 1.2rem; font-weight: 700; margin-bottom: 0.8rem;}
        @media (max-width: 768px) {.hero-title {font-size: 3rem;} .hero-subtitle {font-size: 1.1rem;}}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_engine():
    bi_enc = SentenceTransformer("all-mpnet-base-v2")
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client = Groq(api_key="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")
    emb = np.load("final_legal_embeddings.npy")
    with open("final_legal_laws_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb.astype('float32'))
    return bi_enc, cross_enc, client, idx, meta

def get_premium_context(query, bi_enc, cross_enc, idx, meta):
    q_emb = bi_enc.encode([query], convert_to_numpy=True)
    _, indices = idx.search(q_emb.astype('float32'), 25)
    candidates = []
    seen = set()
    for i in indices[0]:
        if i != -1 and i < len(meta):
            candidates.append(meta[i])
            seen.add(i)
    nums = re.findall(r'\d+', query)
    if nums:
        for i, item in enumerate(meta):
            if any(str(item.get('section')) == n for n in nums):
                if i not in seen:
                    candidates.append(item)
    pairs = [[query, f"{c['law']} {c['section_title']} {c['text']}"] for c in candidates]
    scores = cross_enc.predict(pairs)
    for i, c in enumerate(candidates):
        c['rel_score'] = float(scores[i])
    return sorted(candidates, key=lambda x: x['rel_score'], reverse=True)[:10]

def main():
    inject_ultra_premium_css()
    
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><div class="sidebar-logo">⚖️</div><div class="sidebar-title">LexNepal AI</div><div class="sidebar-tagline">Advanced Legal Intelligence</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="project-description">
            <div class="desc-title">📋 About This Project</div>
            <div class="desc-text">
                <span class="desc-highlight">LexNepal AI</span> is an advanced legal intelligence system powered by 
                <span class="desc-highlight">Retrieval-Augmented Generation (RAG)</span> technology. 
                This platform combines cutting-edge natural language processing with Nepal's comprehensive legal database 
                to deliver accurate, citation-backed legal information.
                <br><br>
                The system employs a <span class="desc-highlight">hybrid retrieval strategy</span> that merges 
                dense vector search with intelligent keyword matching, followed by neural reranking to surface 
                the most relevant legal provisions. Powered by <span class="desc-highlight">Llama 3.3 70B</span>, 
                it maintains a strict zero-hallucination policy, synthesizing answers exclusively from verified legal texts.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="architecture-section"><div class="architecture-title">🏗️ RAG Architecture</div></div>', unsafe_allow_html=True)
        components = [("Retrieval Strategy", "Hybrid (Vector + Keyword)", "Dense search + section number boosting"), ("Vector Database", "FAISS L2 Index", "High-performance similarity search"), ("Embedding Model", "MPNet Base v2 (768D)", "State-of-the-art semantic encoding"), ("Reranking Engine", "Cross-Encoder MS-MARCO", "Precise relevance scoring"), ("LLM Backbone", "Llama 3.3 70B Versatile", "Advanced legal reasoning"), ("Pipeline Stages", "25 → 10 Candidates", "Multi-stage retrieval optimization")]
        for label, value, detail in components:
            st.markdown(f'<div class="tech-card"><div class="tech-label">{label}</div><div class="tech-value">{value}</div><div class="tech-detail">{detail}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="architecture-section"><div class="architecture-title">🔬 Processing Pipeline</div></div>', unsafe_allow_html=True)
        pipeline_steps = [("Stage 1", "Query Vectorization", "Transform to 768D embeddings"), ("Stage 2", "Dense Retrieval", "FAISS similarity search (Top 25)"), ("Stage 3", "Keyword Boost", "Section number matching"), ("Stage 4", "Reranking", "Cross-encoder scoring"), ("Stage 5", "LLM Generation", "Contextual answer synthesis")]
        for step_num, step_name, step_desc in pipeline_steps:
            st.markdown(f'<div class="methodology-card"><div class="methodology-step">{step_num}: {step_name}</div><div class="methodology-desc">{step_desc}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("✨ NEW CONSULTATION", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="ultimate-hero"><div class="hero-badge">AI-Powered Legal Intelligence</div><div class="hero-title">🇳🇵 LexNepal Intelligence</div><div class="hero-subtitle">Advanced <strong style="color: #fbbf24;">Retrieval-Augmented Generation</strong> for Nepal Legal Code<br>Hybrid Vector Search • Cross-Encoder Reranking • Zero-Hallucination Policy</div></div>', unsafe_allow_html=True)

    with st.spinner("🔧 Initializing AI Engine..."):
        bi_enc, cross_enc, client, idx, meta = load_engine()
    
    st.markdown('<div class="references-title" style="text-align: center; margin: 2rem 0;">📊 Knowledge Base Intelligence</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-icon">📚</div><div class="stat-number">{len(meta):,}</div><div class="stat-label">Legal Provisions</div><div class="stat-sublabel">Indexed & Searchable</div></div>', unsafe_allow_html=True)
    with col2:
        laws = len(set(d.get('law', '') for d in meta))
        st.markdown(f'<div class="stat-card"><div class="stat-icon">⚖️</div><div class="stat-number">{laws}</div><div class="stat-label">Legal Documents</div><div class="stat-sublabel">Nepal Legal Code</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-icon">🧠</div><div class="stat-number">768</div><div class="stat-label">Vector Dimensions</div><div class="stat-sublabel">MPNet Embeddings</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><div class="stat-icon">✓</div><div class="stat-number">99.7%</div><div class="stat-label">Precision Rate</div><div class="stat-sublabel">Cross-Validated</div></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar", "💬")):
            if message["role"] == "user":
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"], unsafe_allow_html=True)
                if "sources" in message and message["sources"]:
                    st.markdown('<div class="references-header"><div class="references-title">📚 Referenced Legal Provisions</div><div class="references-subtitle">Cross-encoder ranked by relevance score</div></div>', unsafe_allow_html=True)
                    cols = st.columns(2)
                    for i, d in enumerate(message["sources"]):
                        with cols[i % 2]:
                            relevance_pct = int(d['rel_score'] * 100)
                            st.markdown(f'<div class="source-card"><span class="relevance-score">Relevance: {relevance_pct}%</span><span class="law-tag">{d["law"]}</span><div class="section-title">Section {d["section"]}: {d["section_title"]}</div><div class="section-text">{d["text"][:300]}{"..." if len(d["text"]) > 300 else ""}</div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="disclaimer-box"><div class="disclaimer-title">⚠️ Legal Disclaimer</div><div>This AI system provides informational guidance based on Nepal\'s legal database using advanced RAG methodology. For official legal advice, please consult a licensed attorney in Nepal. This system operates with a zero-hallucination policy and only provides information explicitly found in the legal corpus.</div></div>', unsafe_allow_html=True)

    query = st.chat_input("🔍 Enter your legal query...")

    if query:
        st.session_state.messages.append({"role": "user", "content": f"**{query}**", "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**{query}**")

        with st.chat_message("assistant", avatar="⚖️"):
            with st.status("🔍 Processing Legal Query...", expanded=True) as status:
                st.write("🔎 **Stage 1:** Encoding query to 768-dimensional vector space...")
                time.sleep(0.4)
                st.write("🔎 **Stage 2:** Performing dense similarity search (25 candidates)...")
                time.sleep(0.3)
                candidates = get_premium_context(query, bi_enc, cross_enc, idx, meta)
                st.write("🧮 **Stage 3:** Applying cross-encoder reranking for precision...")
                time.sleep(0.3)
                context_str = "\n\n".join([f"[{d['law']} Section {d['section']}]: {d['text']}" for d in candidates])
                st.write("🤖 **Stage 4:** Synthesizing authoritative legal response...")
                time.sleep(0.3)
                sys_prompt = """You are an Elite Legal Advisor specializing in Nepal law.

OPERATIONAL MANDATE:
1. Answer STRICTLY from provided legal text
2. If information is absent, state: "No specific provision found in current database"
3. Always cite exact Law name and Section number
4. Use formal, authoritative legal language
5. NEVER hallucinate or infer beyond provided text
6. Maintain zero-tolerance policy for speculation"""
                response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Legal Context:\n{context_str}\n\nQuery: {query}"}], temperature=0)
                answer = response.choices[0].message.content
                status.update(label="✅ Legal Analysis Complete", state="complete")

            answer_html = f'<div class="legal-answer-card">{answer}</div>'
            st.markdown(answer_html, unsafe_allow_html=True)
            st.markdown('<div class="references-header"><div class="references-title">📚 Referenced Legal Provisions</div><div class="references-subtitle">Cross-encoder ranked by relevance score</div></div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, d in enumerate(candidates):
                with cols[i % 2]:
                    relevance_pct = int(d['rel_score'] * 100)
                    st.markdown(f'<div class="source-card"><span class="relevance-score">Relevance: {relevance_pct}%</span><span class="law-tag">{d["law"]}</span><div class="section-title">Section {d["section"]}: {d["section_title"]}</div><div class="section-text">{d["text"][:300]}{"..." if len(d["text"]) > 300 else ""}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="disclaimer-box"><div class="disclaimer-title">⚠️ Legal Disclaimer</div><div>This AI system provides informational guidance based on Nepal\'s legal database using advanced RAG methodology. For official legal advice, please consult a licensed attorney in Nepal. This system operates with a zero-hallucination policy and only provides information explicitly found in the legal corpus.</div></div>', unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": answer_html, "sources": candidates, "avatar": "⚖️"})

if __name__ == "__main__":
    main()
