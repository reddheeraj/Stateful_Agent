import streamlit as st
from agent.core import StatefulAgent
import os
from datetime import datetime

def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = StatefulAgent("first_agent")
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'activities' not in st.session_state:
        st.session_state.activities = []

def log_activity(activity_type: str, content: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.activities.insert(0, {
        'timestamp': timestamp,
        'type': activity_type,
        'content': content
    })

def display_activity():
    st.sidebar.subheader("Agent Activity Log")
    for activity in st.session_state.activities[:20]:  # Show last 20 activities
        with st.sidebar.expander(f"{activity['timestamp']} - {activity['type']}"):
            st.caption(activity['content'])

def main():
    st.title("Stateful Agent")
    initialize_agent()

    # # Split layout into two columns
    # col1, col2 = st.columns([3, 1])

    # with col1:  # Main chat column
    user_input = st.chat_input("Message the agent...")
    if user_input:
        # Add user message to history
        st.session_state.history.append({'role': 'user', 'content': user_input})
        log_activity('user_input', user_input)
        
        # Get agent response
        with st.spinner("Thinking..."):
            response = st.session_state.agent.process_message(user_input)
        
        # Add agent response to history
        st.session_state.history.append({'role': 'assistant', 'content': response})
        log_activity('agent_response', response)

        # Rerun to update display
        st.rerun()

    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # with col2:  # Sidebar activity column
    display_activity()

if __name__ == "__main__":
    main()