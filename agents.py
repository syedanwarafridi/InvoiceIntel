from __future__ import annotations
from typing import TypedDict, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------
# 1. Define the shared state
# ---------------------------------------------------------------------
class InvoiceState(TypedDict):
    raw_text: str 
    parsed_fields: Dict[str, Any] | None
    anomaly_report: str | None            
    output: Dict[str, Any] | None


# ---------------------------------------------------------------------
# 2. Configure Gemini LLM (hard‑coded key for demo)
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyBHqGVKVmDFQpHaBolmfrEFJFGtSRrLp9c",
    temperature=0.1,
    streaming=True,
)

# ---------------------------------------------------------------------
# 3. Node functions
# ---------------------------------------------------------------------
def parse_fields(state: InvoiceState) -> InvoiceState:
    """Use Gemini to extract structured data from the raw OCR text."""
    prompt = (
        "You are an AP assistant. Extract the following fields as minified JSON: "
        "vendor, invoice_no, date, currency, total, "
        "line_items (list of {description, qty, unit_price, line_total}).\n\n"
        f"Invoice text:\n{state['raw_text']}"
    )
    state["parsed_fields"] = llm.invoke(prompt).content
    return state


def detect_anomalies(state: InvoiceState) -> InvoiceState:
    """Run simple business‑rule checks on the extracted JSON."""
    prompt = (
        "Assess this invoice JSON for issues. "
        "Rules: (1) total must equal sum(line_total); "
        "(2) duplicate invoice_no triggers alert (assume no duplicates for demo); "
        "(3) total > 50_000 requires manual approval. "
        "Reply 'OK' if clean, otherwise list problems in markdown bullets.\n\n"
        f"JSON:\n```json\n{state['parsed_fields']}\n```"
    )
    state["anomaly_report"] = llm.invoke(prompt).content
    return state


def build_output(state: InvoiceState) -> InvoiceState:
    """Prepare the final payload for downstream systems."""
    state["output"] = {
        "invoice": state["parsed_fields"],
        "anomaly_report": state["anomaly_report"],
        "status": (
            "needs_review"
            if state["anomaly_report"] and state["anomaly_report"] != "OK"
            else "clean"
        ),
    }
    return state


# ---------------------------------------------------------------------
# 4. Build the LangGraph
# ---------------------------------------------------------------------
graph = StateGraph(InvoiceState)

graph.add_node("parse", parse_fields)
graph.add_node("validate", detect_anomalies)
graph.add_node("format", build_output)

graph.add_edge(START, "parse")

graph.add_edge("parse", "validate")
graph.add_edge("validate", "format")

graph.add_edge("format", END)

invoice_intel = graph.compile()