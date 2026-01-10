# This file handles the streamlit front end

# app.py
import streamlit as st
from predictor import predict_top_3
from gpt_api import fetch_disease_info

st.set_page_config(page_title="Gene-to-Disease Predictor", layout="centered")
st.title("🧬 Gene-to-Disease Predictor")
st.markdown("Enter gene details to get top 3 disease predictions.")

# Inputs
gene_id = st.text_input("🔖 Gene ID")
associated_genes = st.text_area("🧬 Associated Genes (comma-separated)")
related_genes = st.text_area("🔗 Related Genes (comma-separated)")

# Predict Button
if st.button("🚀 Predict"):
    if gene_id and associated_genes and related_genes:
        results = predict_top_3(gene_id, associated_genes, related_genes)
        st.session_state["results"] = results
    else:
        st.warning("Please fill all the fields before submitting.")

# Show predictions if available
if "results" in st.session_state:
    st.subheader("🔍 Top 3 Predicted Diseases:")
    for idx, (disease, prob) in enumerate(st.session_state["results"]):
        st.markdown(f"### 🦠 {disease} (Confidence: {prob:.2%})")
        if st.button(f"ℹ️ Know more about {disease}", key=f"info_{idx}"):
            with st.spinner("Fetching info from GEMINI..."):
                info = fetch_disease_info(disease)
                st.info(info)

