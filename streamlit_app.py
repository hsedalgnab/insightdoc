import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="InsightDoc",
    page_icon="📄",
    layout="centered",
    menu_items={
        "About": "InsightDoc — AI-powered PDF chat application for document analysis."
    }
)

load_dotenv()

st.title("Chat with your PDF 📄")
st.caption("AI can make mistakes. Please double-check responses.")

# Accept PDF upload from the user
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and process the PDF only when a new file is uploaded
    if "retriever" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(chunks, embeddings)
            st.session_state.retriever = db.as_retriever()
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []

    # Initialize the LLM using a custom OAI-compatible endpoint
    llm = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_BASE_URL"),
        model="auto"
    )

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Accept user questions in a loop
    if question := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Retrieve relevant chunks, build prompt, and stream the answer
        relevant_docs = st.session_state.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

        with st.chat_message("assistant"):
            response = st.write_stream(
                chunk.content for chunk in llm.stream(prompt)
            )

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a PDF file to get started.")