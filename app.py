#!/usr/bin/env python
"""
Streamlit front‑end pour DeepResearcher
--------------------------------------

$ pip install streamlit
$ streamlit run app.py
"""

import logging
import streamlit as st

from providers.openai_provider import OpenAIProvider
from providers.ollama_provider import OllamaProvider
from deep_research import DeepResearcher

# ---------- UI ----------
st.set_page_config(page_title="DeepResearcher", layout="wide")
st.title("🔍 Local Deep Researcher")

with st.sidebar:
    st.header("⚙️  Paramètres")
    provider = st.selectbox("LLM provider", ["openai", "ollama"])
    model = st.text_input(
        "Nom du modèle",
        "gpt-4o-mini" if provider == "openai" else "llama3:8b"
    )
    site = st.text_input("Site à crawler (laisser vide pour web)", "")
    max_tasks = st.slider("Max sous‑questions", 1, 6, 3)
    max_iters = st.slider("Max itérations", 1, 4, 2)
    debug = st.checkbox("Verbose debug logs")

question = st.text_input("Question de recherche", "")
run_btn = st.button("🚀 Lancer la recherche")

# ---------- zone logs ----------
log_box = st.empty()

# ---------- exécution ----------
if run_btn and question:
    # configure logging -> stream dans la page
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s | %(message)s"
    )
    logger = logging.getLogger()

    # --- handler pour affichage dans Streamlit
    log_messages = []

    class StreamHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            log_messages.append(msg)
            log_box.text("\n".join(log_messages[-25:]))

    logger.addHandler(StreamHandler())

    # instancie LLM provider
    llm = (OpenAIProvider(model) if provider == "openai"
           else OllamaProvider(model))

    # lance DeepResearch
    dr = DeepResearcher(
        llm, question,
        site or None,
        max_tasks=max_tasks,
        max_iters=max_iters
    )

    with st.spinner("Recherche en cours..."):
        answer = dr.run()

    st.success("Recherche terminée.")
    st.subheader("📝 Synthèse (markdown)")
    st.markdown(answer, unsafe_allow_html=True)