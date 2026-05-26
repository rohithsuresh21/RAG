# file loader
from pyexpat import model

from langchain_community.document_loaders import ( 
    PyPDFLoader,  
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader
)


loader = PyPDFLoader("example.pdf")
docs = loader.load()

# text spliting
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     SpacyTextSplitter,
                                     MarkdownTextSplitter,
                                     TokenTextSplitter
)
splitter  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([doc.page_content for doc in docs])

# tokenizing
from langchain.text_splitter import TokenTextSplitter

tkn_txt_splitter = TokenTextSplitter(
    chunk_size = 500,
    chunk_overlap =50,
    encoding_name = "cl100k_base"
)

# count token
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
n = len(encoding.encode("hello world"))

# Sentence-aware tokenizer:
from langchain.text_splitter import (
SentenceTransformersTokenTextSplitter
)

splitter = SentenceTransformersTokenTextSplitter(
chunk_size = 100,
chunk_overlap = 20,
)

# Embedding 
# 1)openai
from langchain_opensai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small')

# 2)Huggingface
from langchian_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# 3)chohere
from langchain_community.embeddings import CohereEmbeddings

embedding = CohereEmbeddings(model = "small", cohere_api_key = "your_api_key")

# 4)Google
from langchain_community.embeddings import GooglePalmEmbeddings

embedding = GooglePalmEmbeddings(model = "models/embedding-gecko-001", google_api_key = "your_api_key")


# query embed
q_embed = embedding.embed_query("What is RAG?")

# vector database

# via FAISS
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index")  # save the index to disk
vectorstore = FAISS.load_local("faiss_index", embedding)  # load the index from disk

#via chroma
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(chunks, embedding, store_directory = './chroma_db')

#via qdrant
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(chunks, embedding, collection_name = "my_collection", host = "localhost", port = 6333)

#similarity search
results = vectorstore.similarity_search("What is RAG?", k=5)

# with score
results = vectorstore.similarity_search_with_score("What is RAG?", k=5)

#seach by vector in list [tupel[document, float]]
results = vectorstore.similarity_search_by_vector(q_embed, k=5, score_threshold=0.5)


# MMR
results = vectorstore.max_marginal_relevance_search("What is RAG?", k=5, 
                                                    fetch_k=20,            # fetch_k controls the number of initial results to retrieve before applying MMR. A value of 20 means that the top 20 results will be considered for MMR.
                                                    lambda_mult = 0.5      # lambda_mult controls the balance between relevance and diversity in the results. A value of 0.5 means that both relevance and diversity are equally weighted.
                                                    )
# retriever

retriever = vectorstore.as_retriever(
    search_type = "similarity",  # similarity, mmr, or hybrid
    search_kwargs = {"k": 5}     # additional arguments for the search method
)

# invoke the retrievers
