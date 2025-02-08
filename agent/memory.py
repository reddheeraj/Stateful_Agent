import faiss
import json
import os
import base64
import numpy as np
from typing import List, Dict
from agent.llm import OllamaWrapper
from datetime import datetime
import streamlit as st
from agent.logger import AgentLogger

class MemoryManager:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.llm_wrapper = OllamaWrapper()
        self.embeddings = self.llm_wrapper.get_embeddings()
        self.llm = self.llm_wrapper.get_llm()
        self.logger = AgentLogger(agent_id)

        # Memory storage
        self.short_term_memory: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        
        # FAISS indices
        self.short_term_index = faiss.IndexFlatL2(4096) # 4096 is the size of the embeddings for Llama 3.1
        self.long_term_index = faiss.IndexFlatL2(4096)
        
        self._load_state()

    def add_memory(self, experience: str, metadata: Dict = None):
        memory = {
            'experience': experience,
            'metadata': metadata or {},
            'embedding': self._embed(experience)
        }
        memory['metadata']['timestamp'] = datetime.now().isoformat()

        self.logger.log_activity("memory_added", {
            "experience": experience[:100],  # Truncate for logging
            "metadata": metadata
        })
        
        # Add to short-term memory
        self.short_term_memory.append(memory)
        self.short_term_index.add(self._to_numpy(memory['embedding']))
        
        # Summarization trigger
        if len(self.short_term_memory) >= 5:
            self._summarize_memories()

        self._save_state()

    def retrieve_memories(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories based on query
        
        Args:
            query (str): Query text
            k (int): Number of memories to retrieve
                
        Returns:
            List[Dict]: List of relevant memories"""
            
        query_embed = self._embed(query)
        query_embed = self._to_numpy(query_embed)
        memories = []
        
        # Check if indices are empty
        if self.short_term_index.ntotal == 0 and self.long_term_index.ntotal == 0:
            return memories
        
        # Search both indices
        _, short_indices = self.short_term_index.search(query_embed, k)
        _, long_indices = self.long_term_index.search(query_embed, k)
        
        for idx in short_indices[0]:
            if idx != -1:  # Ensure valid index
                memories.append(self.short_term_memory[idx])
        for idx in long_indices[0]:
            if idx != -1:  # Ensure valid index
                memories.append(self.long_term_memory[idx])
        
        st.session_state.activities.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'type': 'memory_retrieval',
            'content': f"Query: {query}\nFound {len(memories)} relevant memories"
        })
            
        return sorted(memories, key=lambda x: x['metadata'].get('importance', 0), reverse=True)[:k]

    def _summarize_memories(self):
        # Generate summary using LLM
        summary = self.llm.invoke(
            f"Summarize these memories while preserving key details:\n" +
            "\n".join([m['experience'] for m in self.short_term_memory])
        ).content
        
        # Create long-term memory entry
        long_term_entry = {
            'experience': summary,
            'metadata': {'type': 'summary', 'source_count': len(self.short_term_memory)},
            'embedding': self._embed(summary)
        }
        
        # Add to long-term storage
        self.long_term_memory.append(long_term_entry)
        self.long_term_index.add(self._to_numpy(long_term_entry['embedding']))
        
        # Clear short-term memory
        self.short_term_memory.clear()
        self.short_term_index.reset()

    def _embed(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def _to_numpy(self, embedding: List[float]):
        return np.array([embedding], dtype=np.float32)

    def _save_state(self):
        os.makedirs('data', exist_ok=True)
        state = {
            'short_term': self.short_term_memory,
            'long_term': self.long_term_memory,
            'short_index': base64.b64encode(faiss.serialize_index(self.short_term_index)).decode('utf-8'),
            'long_index': base64.b64encode(faiss.serialize_index(self.long_term_index)).decode('utf-8')
        }
        with open(f'data/{self.agent_id}_memory.json', 'w') as f:
            json.dump(state, f, default=str)

    def _load_state(self):
        try:
            with open(f'data/{self.agent_id}_memory.json', 'r') as f:
                state = json.load(f)
                
            self.short_term_memory = state['short_term']
            self.long_term_memory = state['long_term']
            
            self.short_term_index = faiss.deserialize_index(
                base64.b64decode(state['short_index'].encode('utf-8'))
            )
            self.long_term_index = faiss.deserialize_index(
                base64.b64decode(state['long_index'].encode('utf-8'))
            )
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading memory state: {e}")
            # Initialize empty indices if loading fails
            self.short_term_index = faiss.IndexFlatL2(4096)
            self.long_term_index = faiss.IndexFlatL2(4096)