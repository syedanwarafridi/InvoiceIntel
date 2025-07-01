import streamlit as st
import tempfile
from pathlib import Path
import base64
import os
import json

from agents import invoice_intel
from utils import extract_text_auto


# ---------------------------- Page Setup -------------------------------------
st.set_page_config(
    page_title="InvoiceIntel",
    page_icon="📥",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("InvoiceIntel - Smart Invoice Extraction & Validation")
st.markdown(
    """
    Welcome to **InvoiceIntel**: An AI-powered app that extracts and validates invoice data 
    from PDF or image files. Upload your file, and let the LangGraph agent process it intelligently.
    """
)


# ---------------------------- Sidebar ----------------------------------------
st.sidebar.header("Settings")
ocr_backend = st.sidebar.selectbox("OCR Engine", ["easyocr", "paddle"])
lang_list = st.sidebar.multiselect("OCR Language(s)", ["en", "fr", "de", "es"], default=["en"])


# ---------------------------- File Upload ------------------------------------
uploaded_file = st.file_uploader("Upload an Invoice (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.success("File uploaded successfully")

    try:
        with st.spinner("Extracting text..."):
            raw_text = extract_text_auto(temp_path, ocr_backend=ocr_backend, languages=lang_list)

        with st.spinner("Running InvoiceIntel agent..."):
            result = invoice_intel.invoke({"raw_text": raw_text})

        # ------------------------ Output Display -------------------------------
        st.subheader("Extracted Invoice Data")
        st.json(result["output"], expanded=True)

        # ------------------------ Mermaid Graph -------------------------------
        with st.expander("Show InvoiceIntel Agent Flowchart"):
            png_bytes = invoice_intel.get_graph().draw_mermaid_png()
            st.image(png_bytes, caption="Agent Workflow", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

else:
    st.info("Please upload a PDF or image to begin.")
