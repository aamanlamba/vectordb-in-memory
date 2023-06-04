# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import os
import textwrap


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")
    #pdf_path = "E:\\Code\\vectorstore-in-memory\\2210.03629.pdf"
    pdf_path = "E:\\Code\\vectorstore-in-memory\\20160328110518251825.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    #print(documents[0])
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents)
   # print(docs[0])
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # Create FAISS vectorstore in memory
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    prompt1 = "Give me a summary of ReAct in 5 sentences"
    prompt2 = "Give me the features of Lifecycle Assessment in 5 sentences"
    res = qa.run(prompt2)
    print (textwrap.fill(res,80))
