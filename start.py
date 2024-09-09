from langchain_community.document_loaders import PyPDFLoader

#preprocessing libraries 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#Embeddings libraries
from sentence_transformers import SentenceTransformer

#Faiss index
import faiss
import numpy as np
import pickle

#implementing RAG using FAISS index
import os

from transformers import pipeline, logging

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings from transformers library
logging.set_verbosity_error()


#Functions

# Define a function to preprocess the text data
# def preprocess(text):
#     # Tokenize the text
#     words = word_tokenize(text.lower())  # Tokenize and lowercase
#     # Remove stop words
#     words = [w for w in words if w not in stopwords.words("english")]
#     # Stem the words
#     words = [stemmer.stem(w) for w in words]
#     # Lemmatize the words
#     words = [lemmatizer.lemmatize(w) for w in words]
#     return words
# Preprocess PDF content to split into individual FAQs
def preprocess_documents(documents):
    faqs = []
    for doc in documents:
        content = doc.page_content
        # Split content by FAQ pattern (e.g., Q1:, A1:, Q2:, A2:, etc.)
        faqs.extend(content.split('\nQ'))
    return faqs


#implementing RAG using FAISS 
# Process a query
def preprocess_query(query):
    # Tokenize and preprocess the query
    processed_query = preprocess(query)
    return " ".join(processed_query)  # Join words to form a single string for embedding

# Load FAISS index
def load_faiss_index(index_path="faiss_index.bin"):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        raise FileNotFoundError("FAISS index file not found.")


# Search for the closest FAQ using cosine similarity
def search_faq_cosine(query, model, documents):
    query_embedding = model.encode([query])
    document_embeddings = [model.encode([doc.page_content])[0] for doc in documents]  # Flatten the embeddings
    similarities = cosine_similarity(query_embedding, document_embeddings)
    most_similar_index = np.argmax(similarities)
    return most_similar_index

# Generate a response using GPT-Neo
def generate_response(faq_answer, user_query):
    generator = pipeline('text-generation', model='distilgpt2')
    prompt = f"FAQ Answer: {faq_answer}\nUser Question: {user_query}\n"
    generated = generator(prompt, max_new_tokens=50, do_sample=True)  # Use max_new_tokens
    return generated[0]['generated_text']


if __name__ == "__main__":
    # Load the PDF file
    pdf_path = r"C:\Users\dhtan\OneDrive\Desktop\CHATBOT\faq.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    #preprocessing PDF file
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()



    # Join the extracted text from all pages
    pdf_text = " ".join([doc.page_content for doc in documents])

    #Load Sentence Transformer
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)



    # Tokenize the text into sentences
    sentences = sent_tokenize(pdf_text)

    # Apply the preprocess function on each sentence
    processed_sentences = [preprocess(sentence) for sentence in sentences]


    # Generate embeddings for the processed sentences
    print("Generating embeddings...")
    sentence_embeddings = model.encode(processed_sentences, convert_to_tensor=False)

    # Create and save FAISS index
    print("Creating and saving FAISS index...")
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])  # Dimension of embeddings
    index.add(sentence_embeddings)  # Add sentence embeddings to index
    faiss.write_index(index, "faiss_index.bin")

    # Save the SentenceTransformer model
    print("Saving SentenceTransformer model...")
    with open("sentence_transformer_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Preprocessing and indexing complete. Files 'faiss_index.bin' and 'sentence_transformer_model.pkl' have been created.")


    #implementing RAG using FAISS 


    # Load the FAISS index
    index = load_faiss_index()

#    Get user input for the query
    query = input("Enter your query: ")

    # Preprocess the query
    processed_query = preprocess_query(query)

    query_embedding = model.encode([preprocess_query(query)], convert_to_tensor=False)


    # Search for the most similar FAQ using cosine similarity
    most_similar_index = search_faq_cosine(query, model, documents)

    # Display the top result
    if most_similar_index < len(documents):
        print(f"Result: Document Index {most_similar_index}")
        print(f"Document Content: {documents[most_similar_index].page_content}")

        #Generate a response using the retrieved FAQ and the user query
        faq_answer = documents[most_similar_index].page_content
        response = generate_response(faq_answer, query)
        print("Response:\n", response)

    else:
        print(f"Error: Document index {most_similar_index} is out of range.")













