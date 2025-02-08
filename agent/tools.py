import requests
import os
from wikipediaapi import Wikipedia
from dotenv import load_dotenv
import serpapi
import streamlit as st
from datetime import datetime
from agent.logger import AgentLogger


load_dotenv()

class WikipediaTool:
    def __init__(self, agent_id: str):
        self.logger = AgentLogger(agent_id)
        self.wiki = Wikipedia(
            user_agent="StatefulAgent/1.0 (https://github.com/reddheeraj)",
            language='en'
        )
    
    def search(self, query: str) -> list:
        """Search Wikipedia and return summaries of relevant pages"""
        self.logger.log_activity("wikipedia_search", {"query": query})
        page = self.wiki.page(query)
        if not page:
            return []
        
        results = page.summary[:1000] # limiting to 1000 characters for now
        st.session_state.activities.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'type': 'wikipedia_search',
            'content': f"Query: {query}\nFound result: {(results[:100])}"
        })
        return results

class WebSearchTool:
    def __init__(self, agent_id: str):
        self.logger = AgentLogger(agent_id)
        self.api_key = os.getenv('SERPAPI_KEY')
    
    def search(self, query, num_results=5):
        self.logger.log_activity("web_search", {"query": query, "num_results": num_results})
        params = {
            'q': query,
            'api_key': self.api_key,
            'engine': 'google',
            'num': num_results
        }
        search = serpapi.search(params)
        results = dict(search)
        # response = requests.get('https://serpapi.com/search', params=params)

        st.session_state.activities.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'type': 'web_search',
            'content': f"Query: {query}\nFound {len(results)} results"
        })
        return self._parse_results(results)

    def _parse_results(self, results):
        return [{
            'title': r.get('title'),
            'link': r.get('link'),
            'snippet': r.get('snippet')
        } for r in results.get('organic_results', [])]