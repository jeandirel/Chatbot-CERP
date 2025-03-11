import os
import sys
import sqlite3
from chromadb.config import Settings
import chromadb

# Définition du répertoire de persistance pour ChromaDB
PERSIST_DIRECTORY = "C:/Users/jenzek/Downloads/Chatbot_M3/M3_db/db_M3_bydoc"

# Utilisation des modèles en local
LLM_MODEL_PATH = "Models/Mistral-7B-Instruct-v0.2-AWQ" 
EMBEDDING_MODEL_NAME = "Models/multilingual-e5-large-instruct"
translation_model = "Models/translation_model"  # Exemple pour un modèle de traduction

# Nombre de threads d'ingestion (peut être modifié selon les performances)
INGEST_THREADS = os.cpu_count() or 8

# Configuration de ChromaDB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Taille du contexte et nombre maximum de tokens
CONTEXT_WINDOW_SIZE = 8192
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE // 2  # Utilisation de "//" pour éviter les nombres flottants

# Paramètres GPU pour AWQ
N_GPU_LAYERS = 50
N_BATCH = 265

# Configuration du modèle
MODEL_ID = "Models/Mistral-7B-Instruct-v0.2-AWQ"
MODEL_BASENAME = "model.safetensors"
