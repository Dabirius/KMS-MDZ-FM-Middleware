# KMS-MDZ-FM-Middleware
Die offizielle Middleware für das Repository für das KI-basierte Wissensmanagement UI des Mittelstand-Digital Zentrums Fokus Mensch. Dieses Repository wurde als Middleware für das Repository [KMS-MDZ-FM](https://github.com/Dabirius/KMS-MDZ-FM.git) entwickelt.

# Document Search & Embedding Service
This repository contains a Flask-based API service designed to process documents, generate embeddings, and store them in a Qdrant vector database for efficient search and retrieval. The service leverages support for custom embedding services and language models that adhere to the OpenAI API format, such as vLLM, as well as OpenAI models.

## Features

- **Document Processing:** Splits text documents into chunks and generates embeddings using a custom service or OpenAI.
- **Qdrant Integration:** Stores and retrieves document embeddings using Qdrant, a high-performance vector database.
- **Language Model Response:** Generates responses based on the content of relevant documents, using an LLM model.
- **Custom Model Support:** Supports custom models that follow the OpenAI API format, making integration straightforward.
- **Custom Embedding Model Support:** Allows integration of custom embedding models, with an example endpoint provided in `multilingual_e5.py`.
- **Logging:** Logs user queries and responses to a file for audit and debugging purposes.
- **Health Check Endpoint:** Provides an endpoint to check the health status of the service.

## Requirements

- Docker

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Dabirius/KMS-MDZ-FM-Middleware.git
   cd KMS-MDZ-FM-Middleware
   ```

2. **Configure Environment Variables:**

   Update the `data.json` file with your API key and other relevant configuration details.

   Below is an example of how the `data.json` file should look with a custom model.

   ### Example `data.json` for Custom Model (such as vLLM API)

   ```json
   {
     "open_ai_api_key": "",
     "llm": {
       "open_ai_llm": false,
       "llm_url": "https://your-custom-llm.com/api/v1/chat/completions",
       "llm_post_body": { // Should be adjusted to your API endpoint. If open_ai_llm is true, model is needed.
         "model": "your-model"
       },
       "llm_post_header": { // Should be adjusted to your API endpoint. If open_ai_llm is true, the bearer is needed.
         "Content-Type": "application/json",
         "Authorization": "Bearer your_custom_api_key"  // Optional, depending on your setup
       },
       "system_prompt": "You are a highly knowledgeable AI assistant.",
       "pre_prompt": "You are an assistant designed to help with document searches. Answer the questions based on the documents."
     },
     "embeddings": {
       "open_ai_embeddings": false, // Set to false for own embedding model
       "embedding_model": "", // Needed when open_ai_embeddings is true
       "embedding_dimension": 3072, // Adjust to the input dimension of your embedding model
       "qdrant_collection": "your_collection_name",
       "url_embeddings": "https://your-custom-embedding-service.com/api/embeddings" // Add own embedding endpoint here if open_ai_embeddings is false. Example endpoint is in multilingual_e5.py
     },
     "search": {
       "top_k": 5 // Can be adjusted, however it depends on the max context length of your LLM.
     }
   }
   ```
   The `url_embeddings` field points to a custom embedding service, such as one implemented in `multilingual_e5.py`.

3. **Run the service using Docker Compose:**

   ```bash
   docker-compose up --build
   ```
   This will build and start the service, making it available at http://localhost:5000.

## Endpoints

- **Health Check:**
  - `GET /health`
  - Returns the health status of the service.

- **Process Text and Store Embeddings:**
  - `POST /text_to_db`
  - Processes incoming text data, splits it, generates embeddings, and stores them in the Qdrant collection.

- **Handle User Messages:**
  - `POST /user_message`
  - Handles user queries, retrieves relevant documents, and generates a response using the language model.

## Example Custom Embedding Service
The `multilingual_e5.py` file contains an example implementation of a custom embedding service.

## Logging
All interactions are logged in the `chat_log.txt` file, and any errors are logged in the `error_log.txt` file.