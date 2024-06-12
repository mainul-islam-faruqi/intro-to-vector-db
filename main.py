import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import VectorDBQA, OpenAI

load_dotenv()
os.environ['PINECONE_API_KEY'] = "4adef164-7ca5-4de6-89ef-faabcd938f77"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if __name__ == '__main__':
    print("Hello VectorStore!")
    loader = DirectoryLoader('/Users/MIF/Desktop/Langchain/intro-to-vector-db/mediumblogs', glob="**/*.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name="medium-blogs-index")

    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    query = "What is a vector DB? Give me a 15 word answer for a begginner"
    result = qa({"query": query})
    print(result)