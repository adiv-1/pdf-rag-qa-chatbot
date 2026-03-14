import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from google import genai
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import time


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 100):
            if i > 0:
                time.sleep(2)
            batch = texts[i:i + 100]
            result = self.client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=batch
            )
            all_embeddings.extend([list(e.values) for e in result.embeddings])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[text]
        )
        return list(result.embeddings[0].values)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name.endswith(".pdf"):
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif pdf.name.endswith(".html"):
            html_content = pdf.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "html.parser")
            text += soup.get_text(separator="\n")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = GeminiEmbeddings(api_key=os.getenv("GEMINI_API_KEY"))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"))
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload your documents and click 'Process' first.")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs 🤖")
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and HTML files here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf", "html"])
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()