from openai import OpenAI

# Connect to your local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Use standard completion endpoint
response = client.completions.create(
    model="microsoft/phi-1_5",
    prompt="Explain the difference between supervised and unsupervised learning.",
    max_tokens=100,
)

print(response.choices[0].text)
