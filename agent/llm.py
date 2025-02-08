from langchain_ollama import OllamaEmbeddings, ChatOllama

class OllamaWrapper:
    def __init__(self, model_name="llama3.1:latest"):
        self.embeddings = OllamaEmbeddings(
            model=model_name
        )
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7
        )
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_llm(self):
        return self.llm