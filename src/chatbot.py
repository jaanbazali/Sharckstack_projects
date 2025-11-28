"""OpenAI API integration."""

import requests
from typing import Optional, Dict
from .config import ChatbotConfig
from .conversation import ConversationManager


class OpenAIChatbot:
    """Handles communication with OpenAI API."""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conversation = ConversationManager()
    
    def _make_api_request(self) -> Optional[Dict]:
        """Make HTTP request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": self.conversation.get_messages(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out. Please try again.")
            return None
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                print("ERROR: Invalid API key.")
            elif response.status_code == 429:
                print("ERROR: Rate limit exceeded.")
            else:
                print(f"ERROR: HTTP error: {e}")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network error: {e}")
            return None
    
    def send_message(self, user_message: str) -> Optional[str]:
        """Send a message and get chatbot response."""
        if not user_message.strip():
            return "Please enter a valid message."
        
        self.conversation.add_user_message(user_message)
        response_data = self._make_api_request()
        
        if response_data is None:
            self.conversation.messages.pop()
            return None
        
        try:
            assistant_message = response_data['choices'][0]['message']['content']
            self.conversation.add_assistant_message(assistant_message)
            return assistant_message
        except (KeyError, IndexError) as e:
            print(f"ERROR: Unexpected API response: {e}")
            return None
    
    def reset_conversation(self) -> None:
        """Reset the conversation."""
        self.conversation.reset()
        print("Conversation has been reset.")
    
    def export_conversation(self, filename: Optional[str] = None) -> None:
        """Export conversation history."""
        try:
            filepath = self.conversation.export_conversation(filename)
            print(f"Conversation exported to: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to export: {e}")