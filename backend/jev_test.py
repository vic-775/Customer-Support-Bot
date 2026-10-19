
from jev import triage, JevError


def test_jev():
    message = (
        "I was charged twice for my subscription, "
        "and I need my money refunded immediately!"
    )

    print("\nTesting Jev decision layer...")
    print(f"Customer message: {message}\n")

    try:
        decision = triage(message)

        print("Jev Response")
        print("-" * 40)
        print(f"Intent:       {decision.intent}")
        print(f"Urgency:      {decision.urgency}")
        print(f"Sentiment:    {decision.sentiment}")
        print(f"Needs human:  {decision.needs_human}")
        # print(f"Confidence:   {decision.confidence}")

        assert decision.intent in {
            "billing", "technical", "account", "shipping", "general"
        }
        assert decision.urgency in {"low", "medium", "high"}
        assert decision.sentiment in {
            "positive", "neutral", "frustrated"
        }
        assert isinstance(decision.needs_human, bool)

        print("\nSUCCESS: Jev returned a valid decision.")

    except JevError as e:
        print(f"\nJev API error: {e}")

    except Exception as e:
        print(f"\nTest failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_jev()
