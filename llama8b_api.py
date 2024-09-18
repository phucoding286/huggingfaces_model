from huggingface_hub import InferenceClient

client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)

messages=[
            {"role": "system", "content": "Bạn là một người thiếu nữ bình thường"},
            {"role": "assistant", "content": "Xin chào >0"},
        ]
while True:
    inp = input("bạn: ")
    print()
    print()
    messages.append({"role": "user", "content": inp})
    output = ""
    for message in client.chat_completion(
	    messages,
	    max_tokens=128,
	    stream=True,
        top_p=0.2):
        output += message.choices[0].delta.content
    print(output)
    print()
    print()
    messages.append({"role": "assistant", "content": output})