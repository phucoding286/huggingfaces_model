from huggingface_hub import InferenceClient

client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "i love you"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")



# import requests

# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })

# print(output)