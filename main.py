import streamlit as st
import os
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.title("📄 RAG Chat App")

api_key = st.text_input("Enter your Mistral API Key", type="password")

if api_key:
    os.environ["MISTRAL_API_KEY"] = api_key

    embedding_model = MistralAIEmbeddings()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    vectorstore = None

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        st.success("File processed successfully")

    query = st.text_input("Ask a question")

    if query and vectorstore:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )

        llm = ChatMistralAI(model="mistral-small-2506")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use only context. If not found, say not available."),
            ("human", "context: {context}\nQuestion: {question}")
        ])

        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt.to_messages())

        st.write("🤖 AI:", response.content)

else:
    st.warning("Enter API key first")
