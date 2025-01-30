import streamlit as st
import os
import chromadb
from vector_store import query_vector_store, process_files, get_answer
from datetime import datetime

persist_directory = "./VectorStore"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

client = chromadb.PersistentClient(path=persist_directory)

st.set_page_config(page_title="VectorMind", layout="wide")

st.title("🧠 VectorMind")
st.markdown("<h3 style='color: #4A90E2'>🔍 AI-Powered Vector Store</h3>", unsafe_allow_html=True)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.2

def save_message(role, content):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "time": timestamp
    })

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['content']}** \n\n *{message['time']}*")

def get_collection_names():
    collections = client.list_collections()
    return [collection.name for collection in collections]

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("", ["📤 Upload & Process", "💬 Query Data"])

if page == "📤 Upload & Process":
    st.header("📥 Upload Files to Vector Store")
    uploaded_files = st.file_uploader("Upload PDFs, DOCX, or TXT files", 
                                    type=["pdf", "docx", "txt"], 
                                    accept_multiple_files=True)
    collection_name = st.text_input("Collection Name", "default_collection")
    
    if st.button("🚀 Process Files"):
        if uploaded_files and collection_name:
            status = process_files(uploaded_files, collection_name)
            st.success(status)
        else:
            st.error("Please upload files and enter a collection name.")

elif page == "💬 Query Data":
    st.header("🔍 Query Vector Store")
    init_session_state()
    
    available_collections = get_collection_names()
    collection_name = st.selectbox("📚 Select Knowledge Base", available_collections)
    
    display_chat()
    
    if prompt := st.chat_input("💭 Ask anything about your data..."):
        save_message("user", prompt)
        
        results = query_vector_store(prompt, collection_name)
        
        with st.spinner("🤔 Thinking..."):
            response = get_answer(prompt, collection_name, 
                                temperature=st.session_state.temperature)
            save_message("assistant", response)
        
        with st.expander("📑 View Source Documents"):
            for idx, res in enumerate(results, 1):
                st.markdown(f"**Source {idx}:** {res}")
        
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Settings")
st.session_state.temperature = st.sidebar.slider(
    "🌡️ Response Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)
