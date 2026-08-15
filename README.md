# YouTube RAG QA

A simple app that takes a YouTube video URL, fetches its transcript, and lets you:

- summarize the video
- ask questions about the video content
- get answers based on the transcript using RAG-style retrieval

## Features

- YouTube transcript extraction
- transcript chunking and indexing with FAISS
- embeddings using Hugging Face
- LLM summarization and Q&A using Groq
- simple Gradio web interface

## Requirements

- Python 3.10+
- A Groq API key
- A Hugging Face token

## Setup

1. Clone the project
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add:

```env
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

## Run

```bash
python ytbot.py
```

Then open the local Gradio URL shown in the terminal.

## Usage

- Paste a YouTube video URL
- Click "Summarize Video" to get a summary
- Ask a question in the text box and click "Ask a question"

## Notes

This project is meant to be a lightweight demo and is easy to extend.
