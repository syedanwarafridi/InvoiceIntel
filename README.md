# InvoiceIntel 📥
**Smart AI Agent for Invoice Extraction & Validation, Powered by LangGraph, Streamlit & Gemini**

---

## The Problem

Accounts and procurement teams waste time manually:
- Extracting invoice data from PDFs or images
- Validating totals, fields, and invoice structure
- Checking for missing or invalid items

This results in **errors, inefficiencies, and high processing costs**.

---

## The Solution

**InvoiceIntel** automates this process using AI:

- Upload an invoice (PDF or image)
- Smart extraction (OCR or direct PDF parsing)
- Validates structure using a **LangGraph agent** powered by **Gemini-Pro**
- Outputs structured JSON + anomaly report
- Displays agent flowchart using Mermaid

---

## Agent Architecture

Powered by [LangGraph](https://github.com/langchain-ai/langgraph):

```mermaid
graph TD
    START --> extract
    extract --> validate
    validate --> format
    format --> END
````

---

## Tech Stack

| Layer       | Technology                                |
| ----------- | ----------------------------------------- |
| UI          | Streamlit (responsive, fast)              |
| Agent       | LangGraph + Gemini-Pro (Google GenAI)     |
| OCR         | EasyOCR / PaddleOCR (no Tesseract needed) |
| PDF Parsing | pdfplumber + pdf2image                    |
| Language    | Python 3.9+                               |

---

## Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pdfplumber pdf2image pillow \
            easyocr paddleocr paddlepaddle \
            langchain langgraph langchain-google-genai python-dotenv
```

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/invoiceintel.git
cd invoiceintel
```

2. **Add your Gemini API Key:**

Create a `.env` file in the root folder:

```
GOOGLE_API_KEY=your_api_key_here
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

4. Open [http://localhost:8501](http://localhost:8501) to use the app.

---

## Project Structure

```
.
├── app.py              # Streamlit UI
├── agent.py            # LangGraph AI agent
├── utils.py            # PDF/OCR extraction logic
├── .env                # Your secret keys (not committed)
├── requirements.txt
├── README.md
```

---

## Sample Invoices

You can use:

* Camera photos (`.jpg`, `.png`)
* Scanned PDFs
* Machine-generated invoices

---

## Made By

[Syed Anwar](https://www.linkedin.com/in/syed-anwar-715a411aa/)
Data Scientist

Let’s connect! Open to collaborations in AI, FinTech, and workflow automation.

---
