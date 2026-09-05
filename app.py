from turtle import up
from urllib import response

from huggingface_hub import upload_file
from nltk import chunk
import streamlit as st
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.1:8b")

st.title("Conversational QA Chatbot")
st.write("Uploads Pdf's and chat with their content")
session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

upload_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files= True)

if upload_files:
    documents = []
    for uploaded_file in upload_files:
        tempPdf = f"./temp.pdf"
        with open(tempPdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)

    ## preprocessing 
    #####
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    #####

    ## history_aware_retriever  
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "Which might reference context in the chat history,"
        "formulate a stand alone question which can be understood"
        "without the chat history. Do not answer the question,"
        "just reformulate if its needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chathistory"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    #####

    ## QA chain  
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following piece of retrieved context to answer "
        "the question. If you dont know the answer say that you dont know. Use three sentence"
        "maxixmum and keep the answer concise. \n \n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chathistory"),
                ("human", "{input}")
            ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    #######

    ## Rag chain

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history, input_messages_key="input",
        history_messages_key="chathistory",
        output_messages_key= "answer"
    )

    user_input = st.text_input("Your question: ")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },
        )
        st.write(st.session_state.store)
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)
    


