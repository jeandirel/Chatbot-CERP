Dans le Bash entrer la commande: source .venv/bin/activate
2) Installer les dependances: pip install -r requirements_24_01.txt
Puis: streamlit run streamlit/chatbot_streamlit_main.py
NB: Dans le fichier streamlit/chatbot_streamlit_main, ne pas oublier d'importer le bon fichier backend 
    exp: - from llm_chroma_main import query_llm  # Import function to query LLM
         - from llm_chroma_k import query_llm
