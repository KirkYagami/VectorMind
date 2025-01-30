import os
import tempfile
import json
import gradio as gr
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import chromadb
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
import uuid
import utils

if not utils.pull_ollama_embed_model():
    utils.force_pull_ollama_embed_model


PERSIST_DIRECTORY = "./VectorStore"
EMBED_MODEL = "nomic-embed-text"

if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

persistent_client = chromadb.PersistentClient(PERSIST_DIRECTORY)

def extract_text_from_pdf(pdf_path):
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_docx(docx_path):
    loader = Docx2txtLoader(docx_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    return ""

def process_files(uploaded_files, collection_name):
    text_data = ""
    temp_dir = tempfile.mkdtemp()
    
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        if file.name.endswith(".pdf"):
            text_data += extract_text_from_pdf(file_path) + "\n"
        elif file.name.endswith(".docx"):
            text_data += extract_text_from_docx(file_path) + "\n"
        elif file.name.endswith(".txt"):
            text_data += extract_text_from_txt(file_path) + "\n"
    
    return store_text_to_vector_db(text_data, collection_name)

def store_text_to_vector_db(text, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    collection = persistent_client.get_or_create_collection(collection_name)

    for i, chunk in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        
        # Generate embedding using Ollama
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=chunk
        )
        embedding = response["embedding"]

        collection.add(
            documents=[chunk],
            metadatas={"chunk_id": i},
            ids=[doc_id],
            embeddings=[embedding]
        )

    return f"Data successfully added to collection: {collection_name}"

def query_vector_store(prompt, collection_name, n_results=5):
    collection = persistent_client.get_or_create_collection(collection_name)

    # Generating embedding for query
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=prompt
    )
    query_embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=n_results
    )
    
    return results["documents"] if results else ["No results found."]


model = "llama3.2"
def generate_response(prompt):
    return ollama.generate(model=model, prompt=prompt)['response']

template = """Use the following context to answer the question. If you don’t know the answer, say you don’t know.
{context}
Question: {question}
Answer:"""

# def get_answer(question, collection_name, temperature=0.1):
#     context = ", ".join(query_vector_store(question, collection_name))
#     return generate_response(template.format(context=context, question=question))

def get_answer(question, collection_name, temperature=0.1):
    context_docs = query_vector_store(question, collection_name)
    context = " ".join(context_docs[0]) if context_docs else ""
    template = "Use the following context to answer the question. If you don't know the answer, say you don't know.\nContext: {context}\nQuestion: {question}\nAnswer:"
    return ollama.generate(
        model=model, 
        prompt=template.format(context=context, question=question)
        # temperature=temperature
    )['response']

def interface():
    with gr.Blocks() as ui:
        with gr.Tab("Upload Files"):
            file_input = gr.File(file_types=[".pdf", ".txt", ".docx"], label="Upload Documents", multiple=True)
            collection_input = gr.Textbox(label="Collection Name", placeholder="Enter collection name")
            submit_button = gr.Button("Process Files")
            output_text = gr.Textbox(label="Status")
            
            submit_button.click(fn=process_files, inputs=[file_input, collection_input], outputs=output_text)
        
        with gr.Tab("Query Vector Store"):
            collection_select = gr.Textbox(label="Collection Name", placeholder="Enter collection to query")
            query_input = gr.Textbox(label="Ask a question")
            query_button = gr.Button("Search")
            query_output = gr.Textbox(label="Results")
            
            query_button.click(fn=query_vector_store, inputs=[query_input, collection_select], outputs=query_output)
        
    ui.launch(share=True)

if __name__ == "__main__":
    interface()
