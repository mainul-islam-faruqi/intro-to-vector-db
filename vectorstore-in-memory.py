import os 
# os.environ['OPENAI_API_KEY'] = 'api key here, using folowing line for hiding api key'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS

if __name__ == '__main__':
    print("hi")
    pdf_path = 'c:\\Users\\MIF\\Desktop\\Langchain\\intro-to-vector-db\\Antonio Melé - Django 3 By Example_ Build powerful and reliable Python web applications from scratch (2020, Packt Publishing Ltd) - libgen.lc.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index_react")

    # new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)

    retriever = vectorstore.as_retriever()

    qa = retriever.invoke("what is Django? Give me a 15 word for bigginner")
    for item in qa:
        print(item.page_content)