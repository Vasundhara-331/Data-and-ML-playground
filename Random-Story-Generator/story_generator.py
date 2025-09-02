pip install transformers

from transformers import pipeline

# Load a small pre-trained model
generator = pipeline("text-generation", model="gpt2")

# Try generating text
output = generator("Once upon a time", max_length=30, num_return_sequences=1)
print(output[0]['generated_text'])
