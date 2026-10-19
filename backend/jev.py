"""
jev.py - Jev decision layer for a customer support bot.
 
Jev is a "System One" model: it does NOT write replies. Takes a state
(the customer's message) plus typed questions, and it returns structured
answers with probabilities. The bot uses those answers to decide what to do,
then a normal chat LLM (Claude, Groq, etc.) writes the actual reply.
"""

import os
import requests
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

# used open code to acess a free jev model
OPENCODE_APIKEY = os.getenv("OPENCODE_APIKEY")
if not OPENCODE_APIKEY:
    raise ValueError("OPENCODE_APIKEY is not set in the environment variables.")

JEV_URL = "https://opencode.ai/inference/systemone/v1/systemone"
JEV_MODEL = "jev-1.13-free"
TIMEOUT = 15  # seconds

# --- The questions we ask Jev about every customer message ------------------
# "choice" = pick exactly one option. Each key is an allowed answer and its
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the customer's main request?",
        "criteria": {
            "billing": "Charges, refunds, invoices, payment problems",
            "technical": "Bugs, errors, something not working",
            "account": "Login, password, profile or subscription changes",
            "shipping": "Order status, delivery, returns",
            "general": "Product questions, how-to, anything else",
        },
    },
    "urgency": {
        "type": "choice",
        "instructions": "How urgent is this message?",
        "criteria": {
            "low": "Casual question, no deadline or harm",
            "medium": "Customer is blocked but can wait a few hours",
            "high": "Money lost, service down, security issue or deadline",
        },
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the customer's emotional state?",
        "criteria": {
            "positive": "Happy or appreciative",
            "neutral": "Matter-of-fact",
            "frustrated": "Annoyed, impatient or angry",
        },
    },
    "needs_human": {
        "type": "choice",
        "instructions": "Should a human agent handle this instead of the bot?",
        "criteria": {
            "yes": "Legal threats, fraud, safety, complex disputes, or the "
                   "customer asks for a person",
            "no": "A bot can reasonably handle it",
        },
    },
}

# Normalized schema for Jev's System One response; isolates the rest of the code from the raw API shape.
# confidence is used for thresholding in route(); raw is kept for debugging.
@dataclass
class Decision:
    intent: str
    urgency: str
    sentiment: str
    needs_human: bool
    confidence: dict = field(default_factory=dict)  # per-question confidence
    raw: dict = field(default_factory=dict)

# Raised for Jev-specific failures.
class JevError(Exception):
    pass

# Low-level HTTP call to Jev's /systemone endpoint; returns the raw JSON response.
def _call_jev(state, questions: dict) -> dict:
    resp = requests.post(
        JEV_URL,
        headers={
            "Authorization": f"Bearer {OPENCODE_APIKEY}",
            "Content-Type": "application/json",
        },
        json={"model": JEV_MODEL, "state": state, "questions": questions},
        timeout=TIMEOUT,
    )
 
    if resp.status_code == 403:
        raise JevError(
            "403 from Zen. The free tier may be restricted to use inside "
            "OpenCode, or your key lacks access. Try the paid jev-1.13 model "
            "(JEV_MODEL=jev-1.13), or TypeSafe / Vercel AI Gateway / "
            "Cloudflare Workers AI instead."
        )
    if not resp.ok:
        raise JevError(f"Jev request failed ({resp.status_code}): {resp.text[:300]}")
 
    return resp.json()

# Sends the customer message (plus recent history) to Jev via _call_jev(),
# parses the raw response, and returns a normalized Decision.
def triage(message: str, history: list[str] | None = None) -> Decision:
    """Classify a customer message in one parallel Jev call."""
    state = {"customer_message": message}
    if history:
        state["recent_conversation"] = history[-6:]  # keep the state small
 
    data = _call_jev(state, QUESTIONS)
 
    # Answers are keyed by the same ids as the questions. The wrapper key may
    # differ between providers, so look for it defensively.
    answers = data.get("answers") or data.get("results") or data
 
    def pick(key: str) -> str:
        return answers[key]["choice"]
 
    def conf(key: str):
        return answers[key].get("confidence")
 
    return Decision(
        intent=pick("intent"),
        urgency=pick("urgency"),
        sentiment=pick("sentiment"),
        needs_human=pick("needs_human") == "yes",
        confidence={k: conf(k) for k in QUESTIONS},
        raw=data,
    )