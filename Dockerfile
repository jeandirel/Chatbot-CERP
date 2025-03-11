# Utiliser l'image Python la plus récente
FROM python:latest

# Définir le répertoire de travail dans le conteneur
WORKDIR /Chatbot_M3

# Copier uniquement les fichiers nécessaires pour installer les dépendances d'abord
COPY requirements_24_01.txt /Chatbot_M3/

# Installer les dépendances dans un environnement virtuel
RUN python -m venv venv \
    && . venv/bin/activate \
    && pip install --no-cache-dir -r requirements_24_01.txt

# Copier le reste de l'application dans le conteneur
COPY . /Chatbot_M3/

# Définir le répertoire de travail final
WORKDIR /Chatbot_M3/streamlit

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Activer l'environnement virtuel et lancer l'application Streamlit
CMD ["/bin/sh", "-c", ". venv/bin/activate && streamlit run chatbot_streamlit_main.py"]