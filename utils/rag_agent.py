import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, List
import streamlit as st


class RetailRAGAgent:
    """
    Conversational agent for retail sales data using LangChain and Gemini.
    """
    
    def __init__(self, api_key: str, df: Optional[pd.DataFrame] = None):
        """
        Initialize the RAG agent with Gemini API and optional DataFrame.
        
        Args:
            api_key: Google API key for Gemini
            df: Optional pandas DataFrame with sales data
        """
        print(f"DEBUG: RetailRAGAgent.__init__ called - df is None: {df is None}")
        self.api_key = api_key
        self.df = df
        self.llm = None
        self.agent = None
        self.conversation_history = []
        
        self._initialize_llm()
        
        # Initialize the pandas agent if dataframe is provided
        if df is not None:
            print(f"DEBUG: DataFrame provided, calling _initialize_agent")
            self._initialize_agent()
        else:
            print("DEBUG: No DataFrame provided, skipping agent initialization")
    
    def _initialize_llm(self):
        """Initialize the Gemini LLM."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=self.api_key,
                temperature=0.7,
                convert_system_message_to_human=True
            )
            print(f"DEBUG: LLM initialized successfully with model gemini-3.6-flash")
        except Exception as e:
            import traceback
            error_msg = f"Error initializing LLM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"DEBUG: {error_msg}")
            st.error(error_msg)
    
    def set_dataframe(self, df: pd.DataFrame):
        """
        Set or update the DataFrame for the agent.
        
        Args:
            df: pandas DataFrame with sales data
        """
        self.df = df
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the pandas DataFrame agent."""
        print(f"DEBUG: _initialize_agent called - df is None: {self.df is None}, llm is None: {self.llm is None}")
        
        if self.df is None or self.llm is None:
            print("DEBUG: Skipping agent initialization - df or llm is None")
            return
        
        try:
            print("DEBUG: Creating pandas dataframe agent...")
            
            # Create a system prompt for concise responses
            system_prompt = """You are a retail data analyst assistant. When answering questions:
1. Be concise and direct - get straight to the answer
2. Use bullet points for lists
3. Avoid showing intermediate steps or code unless specifically asked
4. Focus on key insights and actionable information
5. Keep responses under 3-4 sentences when possible"""
            
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=False,  # Disable verbose output
                agent_type="tool-calling",
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,  # Reduce iterations for faster responses
                prefix=system_prompt
            )
            print("DEBUG: Pandas dataframe agent created successfully")
        except Exception as e:
            import traceback
            error_msg = f"Error initializing agent: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"DEBUG: {error_msg}")
            st.error(error_msg)
    
    def query(self, question: str) -> str:
        """
        Query the agent with a natural language question.
        
        Args:
            question: User's question about the data
            
        Returns:
            Agent's response
        """
        if self.agent is None:
            return "Agent not initialized. Please upload data first."
        
        try:
            # Add to conversation history
            self.conversation_history.append(("user", question))
            
            # Get response from agent
            response = self.agent.invoke(question)
            print(f"DEBUG: Raw agent response type: {type(response)}, value: {str(response)[:200] if response else 'None'}...")
            
            # Extract the response text - handle various response formats
            response_text = ""
            
            if isinstance(response, dict):
                response_text = response.get('output', '')

                # LangChain may return output as text blocks such as:
                # [{'type': 'text', 'text': '...'}]. Extract their text directly.
                if isinstance(response_text, list):
                    text_parts = []
                    for block in response_text:
                        if isinstance(block, dict) and block.get('text'):
                            text_parts.append(str(block['text']))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    response_text = '\n'.join(text_parts)
                elif isinstance(response_text, dict):
                    response_text = response_text.get('text', '')

                # If output is empty, try other common response fields.
                if not response_text:
                    response_text = response.get('result', response.get('answer', ''))
            else:
                response_text = str(response)
            
            # Clean up the response - remove JSON formatting and extra content
            import json
            import re
            
            # Try to parse as JSON if it looks like JSON
            if isinstance(response_text, str) and (response_text.startswith('{') or response_text.startswith('[')):
                try:
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict):
                        response_text = parsed.get('output', parsed.get('text', parsed.get('answer', str(parsed))))
                    elif isinstance(parsed, list) and len(parsed) > 0:
                        response_text = str(parsed[0])
                except:
                    pass  # If JSON parsing fails, keep as is
            
            # Convert to string if it's still not a string
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Clean up extra whitespace and newlines
            response_text = response_text.strip()
            
            # If response is still very long, try to extract just the final answer
            if len(response_text) > 500:
                # Look for common answer patterns
                answer_patterns = [
                    r'(?:Answer|Result|Final answer|Response)[:\s]+(.+?)(?:\n|$)',
                    r'(?:The answer is|The result is)[:\s]+(.+?)(?:\n|$)',
                ]
                for pattern in answer_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE)
                    if match:
                        response_text = match.group(1)
                        break
            
            # Add to conversation history
            self.conversation_history.append(("assistant", response_text))
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.conversation_history.append(("assistant", error_msg))
            return error_msg
    
    def get_conversation_history(self) -> List[tuple]:
        """
        Get the conversation history.
        
        Returns:
            List of (role, message) tuples
        """
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def simple_chat(self, message: str) -> str:
        """
        Simple chat without DataFrame context (for general questions).
        
        Args:
            message: User's message
            
        Returns:
            LLM response
        """
        if self.llm is None:
            return "LLM not initialized. Please check your API key."
        
        try:
            messages = [
                SystemMessage(content="You are a helpful retail analytics assistant. Provide clear, concise, and actionable insights."),
                HumanMessage(content=message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error: {str(e)}"
