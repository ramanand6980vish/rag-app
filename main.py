import streamlit as st
import os
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.title("📄 RAG Chat App")

# 🔑 API Key input
api_key = st.text_input("Enter your Mistral API Key", type="password")

if api_key:
    os.environ["MISTRAL_API_KEY"] = api_key

    # Embedding model
    embedding_model = MistralAIEmbeddings()

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db_from_ui"
        )

        st.success("✅ File uploaded and stored in vector DB")

    # Load DB
    vectorstore = Chroma(
        persist_directory="chroma_db_from_ui",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    # LLM
    llm = ChatMistralAI(model="mistral-small-2506")

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Use only the provided context.
If not found, say: I could not find the answer in the document."""),

        ("human", """context: {context}
Question: {question}""")
    ])

    # Chat input
    query = st.text_input("Ask a question")

    if query:
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt.to_messages())

        st.write("🤖 AI:", response.content)

else:
    st.warning("⚠️ Please enter API key to continue")