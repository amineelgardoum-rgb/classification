FROM python:3.11-slim

# Installer pip et dépendances système nécessaires
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements et installer les packages Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
