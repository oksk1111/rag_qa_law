from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class Embedding:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
    
    
    def make_vectorstore(self, source, target, chunk_size=1000, chunk_overlap=100, add_start_index=True):
        '''
        vectorstore 생성
        '''
        # Load, chunk and index the contents
        loader = TextLoader(source, encoding='utf-8')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=target)
        return self.vectorstore


    def load_vectorstore(self, path):
        '''
        생성된 vectorstore 로드
        '''
        self.vectorstore = Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings())
        return self.vectorstore
    

    def get_retriever(self, k=1):
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
