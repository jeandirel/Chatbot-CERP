from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from utils import SentenceTransformerEmbeddingFunction
from langchain.chains import RetrievalQA
import six
from constants import translation_model
import sys
import sqlite3
sys.modules["sqlite3"] = sqlite3
import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    LLM_MODEL_PATH,
    translation_model,
)

from deep_translator import GoogleTranslator, MyMemoryTranslator
import requests
import os
from langdetect import detect, LangDetectException
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

import json
import uuid

# Initialize memory (tracks "question" and conversation history)
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Stores past interactions
    input_key="question",  # Ensures memory tracks the user input
    return_messages=True
)

class CFG:
    model_name = 'mistral-7B'
    temperature = 0.7
    top_p = 0.95
    repetition_penalty = 1.15    
    split_chunk_size = 800
    split_overlap = 0
    embeddings_model_repo = EMBEDDING_MODEL_NAME    
    k = 3  # Number of retrieved documents

# Load Model
def get_model(model=CFG.model_name):
    print('\nLoading Model:', model)
    model_repo = LLM_MODEL_PATH
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )
    
    max_len = 4096
    return tokenizer, model, max_len

tokenizer, model, max_len = get_model()

# Hugging Face Pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    max_new_tokens=max_len,
    temperature=CFG.temperature,
    top_p=CFG.top_p,
    repetition_penalty=CFG.repetition_penalty,
    truncation=True,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Assume that the Helsinki-NLP/opus-mt-en-fr model is already downloaded.
translator = pipeline(
    "translation_en_to_fr",
    model=translation_model,
    tokenizer=translation_model,
    device=0 if torch.cuda.is_available() else -1
)

# Load Embeddings
embedding_func = SentenceTransformerEmbeddingFunction(
    model_path_or_name=EMBEDDING_MODEL_NAME, device="cpu", normalize_embeddings=True
)

# Load ChromaDB
vectordb = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_func,
    collection_name="modop_doc",
)

retriever = vectordb.as_retriever(k=2)

# Improved system prompt
review_template_str = """Tu es un assistant serviable.
Tu reponds toujour en français.
Si tu ne peux pas répondre en utilisant ta base de données, continue la conversation naturellement en tant qu'assistant intelligent.
Tu dois répondre de manière claire, détaillée et structurée. Tes réponses doivent toujours être en français.

Historique de conversation:
{chat_history}

Contexte:
{context}

Utilisateur: {question}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["chat_history","context", "question"],
    messages=messages,
)

review_chain = LLMChain(
    # {"context": retriever, "question": lambda x: x}
    # | review_prompt_template
    # | llm
    # | memory
    llm=llm,
    prompt=review_prompt_template,
    memory=memory
)

def extract_llm_response(response):
    """
    Extracts only the assistant's response from the structured output.
    - Looks for the assistant's answer after 'Assistant:'
    - Cleans unnecessary metadata or context
    """
    if isinstance(response, dict) and "text" in response:
        response_text = response["text"]
    else:
        print("[ERROR] Unexpected response format! Using fallback empty response.")
        return "Je suis désolé, mais je n'ai pas pu générer de réponse."

    # # Extract only the assistant's response
    # return response_text.split("Assistant:", 1)[1].strip() if "Assistant:" in response_text else response_text
    
    # Find the last occurrence of 'Assistant:'
    if "Assistant:" in response_text:
        return response_text.rsplit("Assistant:", 1)[1].strip()
    else:
        print("[ERROR] Could not find 'Assistant:' in response. Returning full text.")
        return response_text  # Return the full text if the split fails


def ensure_french(text):
    try:
        detected_lang = detect(text)
    except LangDetectException:
        # If detection fails, assume it's not French.
        detected_lang = None

    if detected_lang == 'fr':
        return text
    else:
        print(f"Detected language: {detected_lang}. Translating to French...")
        # Translate the text using the already-downloaded Helsinki-NLP/opus-mt-en-fr model.
        translation = translator(text, max_length=512)[0]['translation_text']
        return translation

def classify_with_llm(question):
    """
    Uses the LLM to classify the user's question as 'database' or 'general'.
    Returns 'database' if the question is about a technical process, data retrieval, 
    or information stored in the system. Otherwise, returns 'general'.
    """
    classification_prompt = f"""
    Tu es un assistant chargé de classifier les questions en deux catégories :
    
    1. 'database' : si la question concerne la recherche d'informations techniques, des processus métiers, 
       des données stockées dans une base de données, ou des informations provenant d'une documentation technique.
       
    2. 'general' : si la question concerne une discussion générale, des questions sur le chatbot lui-même, 
       des interactions conversationnelles sans lien avec des données stockées.

    Classifie la question suivante : "{question}"

    Réponds uniquement par 'database' ou 'general', sans autre texte.
    """

    # Ask the LLM to classify the query
    response = llm.invoke(classification_prompt).strip().lower()

    if response not in ["database", "general"]:
        print(f"[WARNING] Unexpected classification response: {response}. Defaulting to 'general'.")
        return "general"

    print(f"[INFO] Query classified as: {response}")
    return response

def query_llm(question):
    """Handles user queries while keeping track of conversation history."""

    # Detect if this is a new conversation (memory is empty)
    chat_history_messages = memory.load_memory_variables({}).get("chat_history", [])
    is_new_conversation = len(chat_history_messages) == 0

    # Classify the query using the LLM itself
    query_type = classify_with_llm(question)

    # Retrieve relevant database documents only if necessary
    relevant_docs = []
    if query_type == "database":
        print("[INFO] Query classified as database-related. Retrieving context from ChromaDB.")
        relevant_docs = vectordb.similarity_search(question, k=1)
    else:
        print("[INFO] Query classified as general conversation. Skipping database retrieval.")

    # Convert chat history to string
    chat_history_str = "\n".join(
        [msg.content for msg in chat_history_messages]
    ) if chat_history_messages else ""

    # Prepare inputs
    inputs = {
        "chat_history": chat_history_str,
        "context": relevant_docs,
        "question": question
    }

    # Decide whether to use RAG or general LLM mode
    if query_type == "general":
        raw_response = llm.invoke(question)  # General conversation mode
    else:
        raw_response = review_chain.invoke(inputs)  # Use RAG with memory

    # Extract assistant response
    # cleaned_response = extract_assistant_response(raw_response)

    # Save interaction in memory
    # memory.save_context({"question": question}, {"response": cleaned_response})
    
    return raw_response
