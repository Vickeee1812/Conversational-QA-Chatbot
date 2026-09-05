

# Conversational QA Chatbot

A powerful conversational chatbot application that enables users to upload PDF documents and interact with their content through natural language questions. Built with LangChain, Streamlit, and Ollama.

## Features

- 📄 **PDF Document Upload** - Support for multiple PDF file uploads
- 💬 **Conversational AI** - Chat with PDF content using LLM
- 🧠 **Chat History** - Maintains conversation history across sessions
- 🔍 **Semantic Search** - Vector-based document retrieval
- 🔐 **Session Management** - Separate chat sessions for different contexts
- 🚀 **Fast Processing** - Efficient text splitting and chunking

## Requirements

- Python >= 3.13
- Ollama (for local LLM inference)
- See `requirements.txt` for Python dependencies

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Conversational-QA-Chatbot
   ```

2. **Install Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Pull required models:
     ```bash
     ollama pull llama3.1:8b
     ollama pull nomic-embed-text
     ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv:
   ```bash
   uv sync
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add any required API keys or configurations

## Usage

1. **Start the Ollama service**
   ```bash
   ollama serve
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

3. **Interact with the chatbot**
   - Enter a session ID (or use default)
   - Upload one or more PDF files
   - Ask questions about the content
   - Chat history is maintained throughout the session

## Project Structure

```
Conversational-QA-Chatbot/
├── app.py                          # Main Streamlit application
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Python dependencies
├── README.md                        # This file
└── src/
    └── conversational_qa_chatbot/
        ├── __init__.py
        └── tt.ipynb               # Experimental/testing notebook
```

## Architecture

The application follows this workflow:

```
User question
    |
    v
RunnableWithMessageHistory
    | adds chat history
    v
History-aware retriever
    | rewrites question if necessary
    | searches vector database
    v
Retrieved documents
    |
    v
Question-answer chain
    | combines question + history + documents
    | asks the LLM
    v
Final answer
```

### Components

- **Document Loader** - PyPDFLoader for PDF extraction
- **Text Splitter** - RecursiveCharacterTextSplitter with 5000 chunk size
- **Embeddings** - Ollama embeddings using nomic-embed-text model
- **Vector Store** - Chroma for semantic search
- **LLM** - Ollama with llama3.1:8b model
- **Chat Memory** - ChatMessageHistory for session management
- **UI** - Streamlit for interactive web interface

## Dependencies

### Core Libraries
- **langchain** - LLM orchestration framework
- **streamlit** - Web application framework
- **ollama** - Local LLM integration
- **chroma** - Vector database
- **pypdf** - PDF document processing

### ML/Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **tensorflow** - Deep learning framework

See `requirements.txt` for complete list of dependencies.

## Configuration

Key configuration parameters in `app.py`:

- **Text Chunk Size**: 5000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: nomic-embed-text
- **LLM Model**: llama3.1:8b

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

- **vicky** - Initial development

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.