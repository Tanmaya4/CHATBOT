# FAQ Chatbot with RAG Using FAISS, Sentence-Transformers, and GPT-Neo
Overview
This project implements a chatbot that answers user queries based on an FAQ document in a PDF format. It uses a Retrieval-Augmented Generation (RAG) pipeline, leveraging FAISS for efficient FAQ retrieval and GPT-Neo for generating responses. The chatbot loads a PDF file, preprocesses its content, searches for the most relevant FAQ using embeddings and cosine similarity, and generates dynamic answers to the user's questions.

Key Features
PDF FAQ Processing: Automatically splits the content into individual FAQs.
Embeddings with Sentence-Transformers: Generates embeddings for FAQ sentences using the SentenceTransformer model (all-MiniLM-L6-v2).
Efficient Search with FAISS: Finds the most similar FAQ using FAISS for fast nearest-neighbor search.
Generative Response using GPT-Neo: Produces detailed responses by combining the retrieved FAQ with GPT-Neo-based text generation.
Preprocessing: Tokenization, stopword removal, stemming, and lemmatization of the text to prepare it for embedding generation.
How It Works
PDF Loading:

The project uses PyPDFLoader to load and extract text from a PDF document containing FAQs.
Preprocessing:

Text is tokenized and processed using nltk, which includes sentence tokenization, stopword removal, stemming (using PorterStemmer), and lemmatization (using WordNetLemmatizer).
Embedding Generation:

Sentence embeddings are generated for each processed FAQ sentence using the SentenceTransformer model.
FAISS Index:

FAISS is used to create an index of the sentence embeddings, which allows an efficient search for the most similar FAQ when a user submits a query.
Cosine Similarity Search:

The user's query is processed into embeddings, and cosine similarity is used to find the closest match from the FAISS index.
Response Generation:

GPT-Neo (distilgpt2) generates a natural language response by combining the retrieved FAQ answer with the user query.
Installation
Clone this repository:

bash
``` git clone https://github.com/your-repo/faq-chatbot-rag.git ```
```cd faq-chatbot-rag ```
Install the required dependencies:

bash
```pip install -r requirements.txt ```
Download necessary NLTK corpora (if not already installed):

bash
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
Usage
Prepare your PDF file:

Place the PDF containing FAQs in a known location (modify the pdf_path variable in the code to point to your PDF).
Run the script:

bash
```
python chatbot.py
```
Interact with the chatbot:

The chatbot will load the PDF, process it, and create the FAISS index.
Enter your query, and the chatbot will retrieve the most relevant FAQ and generate a response.
Key Components
Preprocessing:

Tokenization, stopword removal, stemming, and lemmatization functions are used to prepare text for embedding generation.
FAISS:

A vector search engine that efficiently searches the embeddings for the nearest neighbor based on cosine similarity.
Sentence Embeddings:

The model sentence-transformers/all-MiniLM-L6-v2 is used to generate dense vector representations of FAQ sentences.
GPT-Neo:

A lightweight GPT model (distilgpt2) that is used to generate conversational responses by augmenting the retrieved FAQ answer.
Files and Directory Structure
plaintext

```
faq-chatbot-rag/
├── chatbot.py                  # Main Python script
├── requirements.txt            # Required dependencies
├── faiss_index.bin             # FAISS index (generated)
├── sentence_transformer_model.pkl  # SentenceTransformer model (generated)
├── your_faq.pdf                # Sample FAQ PDF (replace with your own)
└── README.md                   # This file
```
FAQs
Can I use another SentenceTransformer model? Yes, you can choose any model from the sentence-transformers library by changing the model name in the script.

What if I want to use a different PDF loader? You can switch out PyPDFLoader for another loader, such as PyMuPDF or pdfplumber.

What if the FAQ format in my PDF differs from what's expected? Modify the preprocess_documents function to match your FAQ structure.

Future Improvements
Add support for multi-language queries and documents.
Improve accuracy with fine-tuned GPT models or more advanced generative models.
Provide a web interface using Streamlit or Gradio.
License
This project is licensed under the MIT License.

Acknowledgments
HuggingFace for providing the Sentence-Transformers and GPT-Neo models.
FAISS for the efficient vector search implementation.
NLTK for text preprocessing tools.
