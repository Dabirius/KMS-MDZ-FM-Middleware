from flask import Flask, request, jsonify
from openai import OpenAI
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchAny
import requests
import json
from langchain.text_splitter import CharacterTextSplitter
import time
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://qdrant:6333")

# Load configuration data
with open('data.json', 'r') as file:
    config = json.load(file)

# Initialize OpenAI client if applicable
if config["llm"]["open_ai_llm"] or config["embeddings"]["open_ai_embeddings"]:
    openai_client = OpenAI(api_key=config["open_ai_api_key"])

# Global variables from configuration
collection_name = config["embeddings"]["qdrant_collection"]
embedding_dimension = config["embeddings"]["embedding_dimension"]
url_embeddings = config["embeddings"]["url_embeddings"]

# Initialize Qdrant collection if it doesn't exist
def initialize_qdrant_collection():
    collections = qdrant_client.get_collections()
    collection_exists = any(collection.name == collection_name for collection in collections.collections)
    
    if not collection_exists:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
        )

initialize_qdrant_collection()

# Function to get embedding using a specified service
def get_embedding(user_query, url_embeddings):
    """Fetches embedding for a given user query using a specified embedding service."""
    data = {"Nutzeranfrage": user_query}
    response = requests.post(url_embeddings, json=data)
    
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to log user messages and responses to a file
def log_to_file(user_query, chat_response):
    """Logs the user query and corresponding chatbot response to a file with a timestamp."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.txt', 'a') as log_file:
        log_file.write(f"Time: {current_time}\n")
        log_file.write(f"Nutzeranfrage: {user_query}\n")
        log_file.write(f"Chat Response: {chat_response}\n")
        log_file.write("\n" + "=" * 50 + "\n")  # Separator between logs

# Function to split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=2650,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Function to call LLM API and get the response
def api_call(url, headers, body, system_prompt, user_prompt, pre_prompt):
    """Makes an API call to the LLM with the given parameters and returns the response."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    user_prompt = pre_prompt + user_prompt
    messages.append({"role": "user", "content": user_prompt})
    
    body["messages"] = messages
    response = requests.post(url, headers=headers, data=json.dumps(body))
    response_data = response.json()
    
    full_response = response_data["choices"][0]["message"]["content"]#
    model_name = response_data["model"]
    
    return full_response, model_name

# Route to check the health of the service
@app.route("/health", methods=['GET'])
def health_check():
    llm_status = "OpenAI" if isinstance(openai_client, OpenAI) else "Open Source"
    return f"OK! LLM ist {llm_status}", 200

# Route to process text and store embeddings in the database
@app.route("/text_to_db", methods=["POST"])
def text_to_db():
    """Processes incoming text data, splits it, generates embeddings, and stores it in the Qdrant collection."""
    try:
        request_data = request.get_json()
        
        for document in request_data:
            document_texts = []
            document_titles = []
            document_ids = []
            embedded_docs = []
            
            # Process each page in the document
            for page in document["Text"]:
                page_chunks = text_splitter.create_documents([page])
                
                for chunk in page_chunks:
                    page_text = f"{chunk}\nDokumententitel und Quelle: {document['Dokumententitel']}"
                    document_titles.append(document['Dokumententitel'])
                    document_ids.append(document["Dokumenten_ID"])
                    document_texts.append(page_text)
                    
                    # Generate embeddings using either OpenAI or custom embedding service
                    if config["embeddings"]["open_ai_embeddings"]:
                        response = openai_client.embeddings.create(
                            input=page_text,
                            model=config["embeddings"]["embedding_model"]
                        )
                        embedded_docs.append(response.data[0].embedding)
                    else:
                        query_vector = get_embedding(page_text, url_embeddings)
                        embedded_docs.append(query_vector)
                        time.sleep(0.5)
            
            # Prepare and store points in the Qdrant collection
            points = [PointStruct(id=str(uuid.uuid4()), vector=embedded_docs[i], 
                                  payload={"text": document_texts[i], "document_name": document_titles[i], "document_id": document_ids[i]}) 
                      for i in range(len(embedded_docs))]
            
            # Split points into batches for efficient upserts
            def split_list(input_list, batch_size):
                return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]
            
            batches = split_list(points, 100)
            for batch in batches:
                qdrant_client.upsert(collection_name=collection_name, wait=True, points=batch)
    
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(str(e) + "\n")
        return f"Da hat etwas nicht geklappt. {e}", 400
    
    return jsonify({"msg": "Document processed successfully"}), 200

# Route to handle user messages and generate LLM responses
@app.route("/user_message", methods=["POST"])
def user_message():
    """Handles user messages, retrieves relevant documents, and generates an LLM response."""
    try:
        message_data = request.get_json()
        
        # Generate query embedding using either OpenAI or custom embedding service
        if config["embeddings"]["open_ai_embeddings"]:
            response = openai_client.embeddings.create(
                input=message_data["Nutzeranfrage"],
                model=config["embeddings"]["embedding_model"]
            )
            query_vector = response.data[0].embedding
        else:
            query_vector = get_embedding(message_data["Nutzeranfrage"], url_embeddings)
        
        # Search for similar documents in the Qdrant collection
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="document_id", 
                                     match=MatchAny(any=[i for i in message_data["Dokumenten_IDs"]]))]
            ),
            limit=config["search"]["top_k"]
        )
        
        top_k_texts = [result.payload["text"] for result in results]
        top_k_texts.reverse()
        
        # Construct the final message for LLM
        final_message = "Dokumententexte:\n"
        final_message += "\n".join([f"Dokumententext aus denen du eventuell deine Antwort generieren kannst: {text}\n{'-'*25}" for text in top_k_texts])
        final_message += f"\n{message_data['Nutzeranfrage']}"
        final_message += "\nBeantworte die Anfrage auf Deutsch und nur, wenn du im Kontext Informationen dazu findest. Gib die Quellen für deine Antwort an."
        
        # Call LLM API to get the response
        llm_url = config["llm"]["llm_url"]
        if not llm_url.endswith("v1/chat/completions"):
            llm_url = llm_url + "v1/chat/completions"
        
        chat_response, _ = api_call(
            url=llm_url, 
            headers=config["llm"]["llm_post_header"], 
            body=config["llm"]["llm_post_body"], 
            system_prompt=config["llm"]["system_prompt"], 
            user_prompt=final_message, 
            pre_prompt=config["llm"]["pre_prompt"]
        )
        
        # Log the interaction
        log_to_file(message_data['Nutzeranfrage'], chat_response)
    
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(str(e) + "\n")
        return f"Da hat etwas nicht geklappt. {e}", 400
    
    return chat_response, 200

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
