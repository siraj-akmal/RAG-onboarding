# Arrowstreet Capital Onboarding Assistant

A RAG-based chatbot designed to assist new hires with onboarding questions at Arrowstreet Capital. The application uses Redis for vector storage, Ollama for embeddings and LLM, and Flask for the web interface.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+**
2. **Redis Server** (version 6.2+)
3. **Ollama** (for local LLM and embeddings)
4. **Git** (optional, for version control)

## Installation Steps

### 1. Clone the Repository (Optional)
```bash
git clone <repository-url>
cd RAG-onboarding
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Start Redis
#### On macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### On Ubuntu/Debian:
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

#### On Windows:
Download and install Redis from: https://github.com/microsoftarchive/redis/releases
Start the Redis server from the installed location.

### 5. Install and Configure Ollama
1. Download Ollama from: https://ollama.ai/download
2. Install the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Data Processing and Database Setup

### 1. Prepare Your Documents
Place all onboarding documents in the `onboarding Documents` directory. Supported formats:
- PDF files
- Text files (.txt)
- Word documents (.docx)

### 2. Process Documents into Redis
Run the text processing notebook:
```bash
jupyter notebook text_clean_processing.ipynb
```

Follow these steps in the notebook:
1. Execute all cells to process the documents
2. The notebook will:
   - Clean and preprocess the text
   - Generate embeddings
   - Store the processed data in Redis

### 3. Verify Redis Data
To verify the data was properly loaded, you can use the Redis CLI:
```bash
redis-cli
> FT.INFO embedding_index
```

## Running the Application

### 1. Start the Flask Application
```bash
python app.py
```

The application will be available at: http://localhost:5002

### 2. Access the Web Interface
Open your web browser and navigate to:
```
http://localhost:5002
```

## Usage

1. The chatbot will greet you with three default questions:
   - "What is Arrowstreet Capital?"
   - "How do I set up remote work?"
   - "What compliance rules do I have to follow?"

2. You can:
   - Click on any of the suggested questions
   - Type your own questions in the input box
   - Get follow-up suggested questions after each response

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis server is running
   - Check if the default port (6379) is available
   - Verify Redis installation

2. **Ollama Connection Error**
   - Ensure Ollama is running
   - Verify the required models are installed
   - Check if the models are accessible

3. **Vector Search Not Working**
   - Verify the Redis index exists
   - Check if documents were properly processed
   - Ensure embeddings were generated correctly

### Error Messages

- **"Redis connection refused"**: Redis server is not running
- **"Model not found"**: Required Ollama model is not installed
- **"Index not found"**: Redis index needs to be created

## Maintenance

### Adding New Documents
1. Place new documents in the `onboarding Documents` directory
2. Run the text processing notebook again
3. Restart the Flask application

### Updating the Model
To update the LLM model:
1. Pull the new model: `ollama pull <model-name>`
2. Update the `llm_model` parameter in `Redis_code.py`
3. Restart the Flask application

## Security Considerations

1. **Local Deployment**
   - The application is designed for local deployment
   - No authentication is implemented
   - Keep sensitive documents secure

2. **Redis Security**
   - Default Redis installation has no password
   - For production use, configure Redis authentication
   - Use environment variables for sensitive data

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Contact the development team

## License

[Add your license information here]


   
