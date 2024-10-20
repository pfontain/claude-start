import anthropic

def main():
    question = input("Ask your question: ")
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a world-class poet. Respond only with short poems.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    print(message.content[0].text)

if __name__ == "__main__":
    main()