import re
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from agent.llm import OllamaWrapper
from agent.memory import MemoryManager
from agent.tools import WebSearchTool, WikipediaTool
from datetime import datetime
from agent.logger import AgentLogger

class StatefulAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = MemoryManager(agent_id)
        self.tools = {
            'web': WebSearchTool(agent_id),
            'wikipedia': WikipediaTool(agent_id)
        }
        self.llm_wrapper = OllamaWrapper()
        self.llm = self.llm_wrapper.get_llm()
        self.logger = AgentLogger(agent_id)
        
        self.persona = f"""
        You are an adaptive AI assistant that learns from interactions. 
        Your personality evolves based on your experiences.
        
        Current date: {datetime.now().strftime("%Y-%m-%d")}
        You have access to complete conversation history. 
        When asked about previous interactions, refer to your memory.
        """

    def process_message(self, message: str) -> str:
        self.logger.log_activity("message_received", {"query": message})
        if self._is_about_history(message):
            self.logger.log_activity("history_query", {"query": message})
            return self._handle_history_query(message)
        
        # Retrieve relevant memories
        context_memories = self.memory.retrieve_memories(message)
        context = "\n".join([m['experience'] for m in context_memories])
        

        search_type = self._determine_search_needs(message)
        if search_type == 'llm':
            # LLM-based response
            response = self.llm.invoke(message).content
            self.memory.add_memory(
                experience=f"User: {message}\nDo not ask back any questions, just answer.\n\nAssistant: {response}",
                metadata={'type': 'conversation'}
            )
            self.logger.log_activity("response_generated", {"response": response})
            return response
        elif search_type == 'web':
            # Web search decision (just a simple heuristic)
            results = self.tools['web'].search(message)
            context += "\nSearch Results:\n" + "\n".join([f"{r['snippet']} (Link: {r['link']})" for r in results])
        elif search_type == 'wikipedia':
            # Wikipedia search
            context += "\nWikipedia Results:\n"
            keywords = self._break_down_query_into_keywords(message)
            for keyword in keywords:
                res = self.tools['wikipedia'].search(keyword)
                context += res
        elif search_type == 'both':
            # Comprehensive search
            context += "\nWikipedia Results:\n"
            web_results = self.tools['web'].search(message)
            keywords = self._break_down_query_into_keywords(message)
            for keyword in keywords:
                res = self.tools['wikipedia'].search(keyword)
                context += res
            context += "\nSearch Results:\n" + "\n".join([r['snippet'] for r in web_results])

        # Generate response
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "Context:\n{context}\n\nQuery: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": message
        }).content
        
        # Update memory
        self.memory.add_memory(
            experience=f"User: {message}\nAssistant: {response}",
            metadata={'type': 'conversation'}
        )
        self.logger.log_activity("response_generated", {"response": response})
        return response
    
    def _break_down_query_into_keywords(self, query: str) -> List[str]:
        prompt = f"""
        Break down the query into keyword or keywords by listing the most important words:
        Query: {query}

        Keywords:

        Respond in format within the triple backticks: ```keyword1, keyword2, keyword3```"""
        
        response = self.llm.invoke(prompt).content
        print("Response: ", response)
        keywords = re.findall(r"```(.*)```", response, re.DOTALL)
        return keywords[0].split(", ")
    
    def _determine_search_needs(self, query: str) -> str:
        """Determine which search tools to use based on query"""
        prompt = f"""Analyze the query and choose search options:
        Query: {query}
        
        Options:
        - 'llm' if the query can be answered without external search.
        - 'web' for general web search
        - 'wikipedia' for factual/encyclopedic info
        - 'both' for comprehensive research
        - 'none' for no search
        
        usually go for both unless the query is very specific.
        If you are unsure or the query looks irrelevant to web search or wikipedia, choose 'llm'.
        Respond in format: 'tool'"""
        response = self.llm.invoke(prompt).content.lower().strip()
        
        self.logger.log_activity("search_decision", {"query": query, "response": response})
        valid_tools = ['both', 'web', 'wikipedia', 'llm']
        for tool in valid_tools:
            if tool in response:
                return tool
        return response if response in valid_tools else 'llm'

    def _is_about_history(self, query: str) -> bool:
        history_keywords = [
            "previous questions", "past conversations", "history",
            "what did I ask", "have we discussed", "before"
        ]
        
        # First check for obvious keywords
        if any(kw in query.lower() for kw in history_keywords):
            return True
        
        history = self.memory.retrieve_memories(query, k=10)
        if not history:
            return False
        # Then use LLM for ambiguous cases
        prompt = f"""Determine if this query is asking about conversation history with an agent:
        Query: {query}
        
        Conversation history:
        {history}
        Respond ONLY with 'yes' or 'no'"""
        
        response = self.llm.invoke(prompt).content.lower().strip()
        return 'yes' in response

    def _handle_history_query(self, query: str) -> str:
        # Retrieve relevant memories
        memories = self.memory.retrieve_memories(query, k=10)
        
        # Filter user questions
        user_questions = [
            m['experience'].split("Assistant:")[0].strip()
            for m in memories 
            if m['metadata'].get('type') == 'conversation'
        ]
        
        # Generate response
        prompt = f"""You're answering a question about conversation history.
        User query: {query}
        
        Relevant previous interactions:
        {chr(10).join(user_questions)}
        
        Respond helpfully using this information."""
        
        response = self.llm.invoke(prompt).content
        
        # Add to memory without creating infinite loop
        self.memory.add_memory(
            experience=f"User: {query}\nAssistant: {response}",
            metadata={'type': 'history_query'}
        )
        
        return response
