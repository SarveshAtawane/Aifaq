from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
import gradio as gr
from google.colab import drive
import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from google.colab import userdata

app = Flask(__name__)

# Specify model HuggingFace model name
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model

# Function for initializing tokenizer
def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

# Load model
model = load_quantized_model(model_name)

# Initialize tokenizer
tokenizer = initialize_tokenizer(model_name)

# Specify stop token ids
stop_token_ids = [0]

# Put the URL here
url = "https://iroha.readthedocs.io/en/develop/index.html"
loader_web = RecursiveUrlLoader(url=url)

# Set the GitHub repo and remember to add your GitHub token to secret keys
GPAT = userdata.get('GITHUB_PERSONAL_ACCESS_TOKEN')
loader_github = GitHubIssuesLoader(
    repo="hyperledger/iroha",
    access_token=GPAT,
)

# Merge the data loaders
loader = MergedDataLoader(loaders=[loader_web, loader_github])
documents = loader.load()

# Filter metadata types fixing an exception
documents = filter_complex_metadata(documents, allowed_types=(str, int, float, bool))

# Split the documents into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)

# Specify embedding model (using HuggingFace sentence transformer)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# Embed document chunks
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# Specify the retriever
retriever = vectordb.as_retriever()

# Build HuggingFace pipeline for using zephyr-7b-alpha
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# Specify the LLM
llm = HuggingFacePipeline(pipeline=pipeline)

# Build conversational retrieval chain with memory (RAG) using LangChain
def create_conversation(query: str, chat_history: list) -> tuple:
    try:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=False
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )
        result = qa_chain({'question': query, 'chat_history': chat_history})
        chat_history.append((query, result['answer']))
        return '', chat_history
    except Exception as e:
        chat_history.append((query, str(e)))
        return '', chat_history

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    chat_history = data.get('chat_history', [])
    _, chat_history = create_conversation(user_input, chat_history)
    chatbot_response = chat_history[-1][1]
    start_text = "Helpful Answer:"
    start_index = chatbot_response.find(start_text)
    response_text = chatbot_response[start_index + len(start_text):] if start_index != -1 else chatbot_response
    return jsonify({'response': response_text, 'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
