from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/create_embedding', methods=['POST'])
def create_embedding():
    data = request.json
    input_text = data["Nutzeranfrage"]

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    # Get embeddings from model
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Ensure the embedding is a NumPy array
    embedding = np.array(embedding)
    
    # Normalize the embedding
    norm_embedding = embedding / np.linalg.norm(embedding)

    return jsonify({"embedding": norm_embedding.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
