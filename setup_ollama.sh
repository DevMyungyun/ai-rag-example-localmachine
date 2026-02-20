#!/bin/bash
# Setup script for Ollama models

echo "Setting up Ollama..."
echo ""

# Load environment variables from .env or .env.example
if [ -f .env ]; then
    source .env
    echo "Loaded configuration from .env"
elif [ -f .env.example ]; then
    source .env.example
    echo "Loaded configuration from .env.example"
fi

# Default model if not set in env
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:7b}

# Check if Ollama container is running
if ! docker ps | grep -q ollama-rag; then
    echo "Starting Ollama container..."
    docker-compose up -d ollama
    sleep 5
fi

echo "Pulling $OLLAMA_MODEL model (this may take a few minutes)..."
docker exec -it ollama-rag ollama pull $OLLAMA_MODEL

echo ""
echo "Available models:"
docker exec -it ollama-rag ollama list

echo ""
echo "✓ Setup complete!"
echo ""
echo "You can now use the RAG system:"
echo "  python rag_query.py                    # Interactive mode"
echo "  python rag_query.py -q 'your question' # Single question"
echo ""
echo "To pull other models:"
echo "  docker exec -it ollama-rag ollama pull llama3.2:3b"
echo "  docker exec -it ollama-rag ollama pull gemma2:9b"
echo "  docker exec -it ollama-rag ollama pull mistral"
