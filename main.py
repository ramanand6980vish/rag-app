import streamlit as st
import os

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.title("📄 RAG Chat App")

api_key = st.text_input("Mistral API Key", type="password")

if api_key:
    os.environ["MISTRAL_API_KEY"] = api_key

    embedding = MistralAIEmbeddings(api_key=api_key)

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:
        with open("temp.pdf", "wb") as f:
            f.write(file.read())

        docs = PyPDFLoader("temp.pdf").load()
        chunks = RecursiveCharacterTextSplitter(500, 50).split_documents(docs)

        db = Chroma.from_documents(chunks, embedding)

        st.success("PDF processed!")

        query = st.text_input("Ask question")

        if query:
            retriever = db.as_retriever(search_kwargs={"k": 4})

            context = "\n\n".join(
                [d.page_content for d in retriever.invoke(query)]
            )

            llm = ChatMistralAI(model="mistral-small-2506", api_key=api_key)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer only from context. If missing, say not found."),
                ("human", "context:{context}\nquestion:{question}")
            ])

            res = llm.invoke(
                prompt.invoke({"context": context, "question": query}).to_messages()
            )

            st.write("🤖", res.content)

else:
    st.warning("Enter API key")
