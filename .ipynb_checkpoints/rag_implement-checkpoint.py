import ollama
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  
from langchain_community.embeddings import HuggingFaceEmbeddings


persist_directory = "vector_db1"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)

custom_pdf_prompt = PromptTemplate.from_template(
    template="""
You are a helpful and concise assistant. Use the following context to answer the user's question.
If the answer isn't in the context, you may provide a related response based only on the PDF,
but refrain from using any outside knowledge or making assumptions beyond the PDF content.

Context: {context}

User's Question: {question}

Answer (based solely on the PDF content):
"""
)

llm = Ollama(model="llama3.2")
retriever = vectorstore.as_retriever(search_kwargs={"k":5})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": custom_pdf_prompt})




