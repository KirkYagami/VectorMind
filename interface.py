import streamlit as st
import os
import chromadb
from vector_store import query_vector_store, process_files

persist_directory = "./VectorStore"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

client = chromadb.PersistentClient(path=persist_directory)

st.set_page_config(page_title="Custom AI Vector Store", layout="wide")
st.title("📚 AI-Powered Vector Store")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Process Data", "Query Vector Store"])




import datetime

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.2

def save_message(role, content):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": role, "content": content, "time": timestamp})


if page == "Upload & Process Data":
    st.header("Upload Files to Vector Store")
    uploaded_files = st.file_uploader("Upload PDFs, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    collection_name = st.text_input("Collection Name", "default_collection")
    
    if st.button("Process Files"):
        if uploaded_files and collection_name:
            status = process_files(uploaded_files, collection_name)
            st.success(status)
        else:
            st.error("Please upload files and enter a collection name.")

elif page == "Query Vector Store":
    st.header("Query the Vector Store")
    collection_name = st.text_input("Collection Name to Query", "default_collection")
    user_query = st.text_area("Ask a question about your data")
    
    if st.button("Search"):
        init_session_state()
        if user_query and collection_name:
            results = query_vector_store(user_query, collection_name)
            st.write("### Results:")
            for res in results:
                st.write(f"- {res}")

            for message in st.session_state.messages:
                message_class = "user-message" if message["role"] == "user" else "assistant-message"
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <div class="message-header">
                            <span>{message["role"].title()}</span>
                            <span>{message["time"]}</span>
                        </div>
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)


            with st.form(key="chat_form", clear_on_submit=True):                
                import vector_store
                if st.form_submit_button("Send"):
                    save_message("user", user_query)
                    with st.spinner("Thinking..."):
                        response = vector_store.get_answer(user_query, temperature=0.1)
                        save_message("assistant", response)
                    st.rerun()
        else:
            st.error("Please enter a query and a collection name.")
