"""Configuration management for the chatbot."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ChatbotConfig:
    """Configuration management for the chatbot."""
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4")
        self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
        self.timeout = 30
        
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Please create a .env file with your API key.")
            return False
        return True