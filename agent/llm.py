from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import Config

class OllamaWrapper:
    def __init__(self):
        self.config = Config()
        self.embeddings = OllamaEmbeddings(
            model=self.config.model,
        )
        self.llm = ChatOllama(
            model=self.config.model,
            temperature=0.7
        )
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_llm(self):
        return self.llm