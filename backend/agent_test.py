from agent import handle_message, ESCALATION_MESSAGE


def test_agent():
    messages = [
        # "How do I change my profile picture?",
        "I was charged twice for my subscription, "
        "and I need my money refunded immediately!",
    ]

    print("\nTesting support agent...")

    for message in messages:
        print(f"\nCustomer message: {message}\n")

        try:
            reply = handle_message(message)

            print("Agent Response")
            print("-" * 40)
            print(reply)
            print(f"\nEscalated:    {reply == ESCALATION_MESSAGE}")

            assert isinstance(reply, str)
            assert len(reply.strip()) > 0

            print("\nSUCCESS: Agent returned a valid reply.")

        except Exception as e:
            print(f"\nTest failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_agent()