from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from utils import SentenceTransformerEmbeddingFunction
from langchain.chains import RetrievalQA
import sqlite3
import sys
sys.modules["sqlite3"] = sqlite3
import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    LLM_MODEL_PATH,
    translation_model,
)

from deep_translator import GoogleTranslator, MyMemoryTranslator
import os
from langdetect import detect, LangDetectException
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Configuration class
class CFG:
    model_name = 'mistral-8B'
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
    max_new_tokens=MAX_NEW_TOKENS,
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

retriever = vectordb.as_retriever(k=CFG.k)

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

# Initialize memory (tracks "question" and conversation history)
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Stores past interactions
    input_key="question",  # Ensures memory tracks the user input
    return_messages=True
)

review_chain = LLMChain(
    llm=llm,
    prompt=review_prompt_template,
    memory=memory
)

def clean_response(response):
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

def query_llm(question):
    """Decides whether to use the RAG model or pure LLM for conversation."""
    relevant_docs = vectordb.similarity_search(question, k=1)
    
    # Detect if this is a new conversation (memory is empty)
    chat_history_messages = memory.load_memory_variables({}).get("chat_history", [])
    
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

    if not relevant_docs:
        print("[INFO] No relevant documents found. Using general chatbot mode.")
        raw_response = llm.invoke(question)  # Pure LLM response
    else:
        print("[INFO] Retrieved relevant documents. Using RAG mode.")
        raw_response = review_chain.invoke(inputs)  # Use retrieval-augmented response

    cleaned_response = clean_response(raw_response)

    print("[INFO] Cleaning and splitting response for translation...")

    final_text = ensure_french(cleaned_response)
    return final_text