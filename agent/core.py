import re
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from agent.llm import OllamaWrapper
from agent.memory import MemoryManager
from agent.tools import WebSearchTool, WikipediaTool
from datetime import datetime
from agent.logger import AgentLogger
from config import Config
import json

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
        self.config = Config()
        
        self.persona = f"""
        You are an adaptive AI assistant that learns from interactions. 
        Your personality evolves based on your experiences.
        
        Current date: {datetime.now().strftime("%Y-%m-%d")}
        You have access to complete conversation history. 
        When asked about previous interactions, refer to your memory.
        """

    def process_message(self, message: str) -> str:
        self.logger.log_activity("message_received", {"query": message})

        if self._is_complex_query(message):
            self.logger.log_activity("complex_query", {"query": message})
            return self._process_complex_query(message)
        
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
            response = self._postprocess_response(response)
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
        
        response = self._postprocess_response(response)
        # Update memory
        self.memory.add_memory(
            experience=f"User: {message}\nAssistant: {response}",
            metadata={'type': 'conversation'}
        )
        self.logger.log_activity("response_generated", {"response": response})
        return response
    
    def _process_complex_query(self, query: str) -> str:
        """Handle complex queries by breaking them into sub-queries"""
        sub_queries = self._decompose_query(query)
        results = []
        
        for sub_q in sub_queries:
            results.append(self._determine_search_needs(sub_q))
        
        return self._synthesize_results(query, sub_queries, results)
    
    def _decompose_query(self, query: str) -> List[str]:
        """Break down complex query into sub-questions"""
        self.logger.log_activity("query_decomposition", {"query": query})
        prompt = f"""Break this complex query into standalone sub-questions:
        Query: {query}
        
        Respond with list in format: ```json
        {{"sub_questions": ["q1", "q2", "q3"]}}```"""
        
        response = self.llm.invoke(prompt).content
        response = self._postprocess_response(response)
        try:
            match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))['sub_questions']
            return [query]
        except:
            return [query]


    def _synthesize_results(self, original_query: str, sub_queries: List[str], results: List) -> str:
        """Combine sub-query results into final answer"""
        context = []
        for q, res in zip(sub_queries, results):
            context.append(f"Sub-query: {q}\nResults: {str(res)[:200]}...")
        
        prompt = f"""Synthesize this information into a coherent answer:
        Original query: {original_query}
        
        Sub-query results:
        {chr(10).join(context)}
        
        Provide a comprehensive answer that addresses all aspects of the original query."""
        res = self.llm.invoke(prompt).content
        res = self._postprocess_response(res)
        self.logger.log_activity("synthesis", {"query": original_query, "response": res})
        self.memory.add_memory(
            experience=f"User: {original_query}\nAssistant: {res}",
            metadata={'type': 'synthesis'}
        )
        return res
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query requires decomposition"""
        prompt = f"""Does this query contain multiple independent questions or require multiple information sources?
        Query: {query}
        
        Consider complex if:
        - Asks about different topics
        - Requires both factual and current information
        - Uses conjunctions like 'and', 'also', 'plus'
        
        Respond ONLY with 'yes' or 'no'"""
        
        response = self.llm.invoke(prompt).content.lower().strip()
        response = self._postprocess_response(response)
        self.logger.log_activity("complex_query_check", {"query": query, "response": response})
        return 'yes' in response
    
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
        Choose web if query is searchable on the web.
        Choose wikipedia if query has keywords that can be searched on wikipedia.
        If you are unsure or the query looks irrelevant to web search or wikipedia, choose 'llm'.
        Respond in format: 'tool'"""
        response = self.llm.invoke(prompt).content.lower().strip()
        response = self._postprocess_response(response)
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
        response = self._postprocess_response(response)
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
        search_type = self._determine_search_needs(query)
        context = "\n".join(user_questions)
        
        # Perform search if needed
        if search_type not in ['llm', 'none']:
            search_results = []
            
            if search_type in ['web', 'both']:
                web_results = self.tools['web'].search(query)
                search_results.extend([f"Web: {r['snippet']}" for r in web_results])
                
            if search_type in ['wikipedia', 'both']:
                wiki_results = []
                keywords = self._break_down_query_into_keywords(query)
                for keyword in keywords:
                    wiki_results.append(self.tools['wikipedia'].search(query))
                search_results.extend([f"Wikipedia: {wiki_results}"])
            
            context += "\n\nFresh Search Results:\n" + "\n".join(search_results)

        # Generate response
        prompt = f"""You're answering a question that combines history and current information.
        User query: {query}
        
        Context from history:
        {context}
        
        If using search results, verify facts with the context. Respond helpfully:"""
        
        response = self.llm.invoke(prompt).content
        response = self._postprocess_response(response)
        
        self.memory.add_memory(
            experience=f"User: {query}\nAssistant: {response}",
            metadata={'type': 'history_query'}
        )
        return response
    
    def _postprocess_response(self, content):
        if self.config.model == "deepseek-r1:14b":
            # Remove everything between <think> tags
            cleaned = re.sub(r'<think>(.*?)(</think>)', '', content, flags=re.DOTALL)
            return cleaned.strip()
        return content

