"""agent.py - Simple support agent: Jev classifies, Groq writes the reply."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from jev import triage, JevError

load_dotenv()

groq_api_key = os.getenv("GROQ_APIKEY")
if not groq_api_key:
    raise ValueError("GROQ_APIKEY is not set in the environment variables.")

# Verify the model name against Groq's current model list.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.3,
    max_tokens=512,
)

ESCALATION_MESSAGE = (
    "I'm sorry for the trouble. I'm connecting you with a member of our "
    "support team who will help you shortly."
)


def handle_message(message: str) -> str:
    # 1. Classify with Jev. If it fails, hand off to a human.
    try:
        decision = triage(message)
    except Exception:
        return ESCALATION_MESSAGE

    # 2. Escalate serious cases.
    if decision.needs_human or (
        decision.urgency == "high" and decision.sentiment == "frustrated"
    ):
        return ESCALATION_MESSAGE

    # 3. Otherwise let Groq write the reply.
    prompt = (
        f"You are a customer support assistant. Issue type: {decision.intent}. "
        f"Customer mood: {decision.sentiment}. Reply briefly and politely. "
        f"Never invent policies or order details.\n\nCustomer: {message}"
    )
    return llm.invoke(prompt).content