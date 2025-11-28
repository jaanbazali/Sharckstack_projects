"""Conversation history management."""

import json
from typing import List, Dict, Optional
from datetime import datetime


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.messages.append({"role": "system", "content": self.system_prompt})
        
    def _default_system_prompt(self) -> str:
        """Returns default system prompt for customer support."""
        return """You are a helpful and professional customer support assistant. 
        Your role is to:
        - Answer customer questions clearly and concisely
        - Be polite, patient, and empathetic
        - Provide accurate information
        - Escalate complex issues when appropriate
        - Maintain a friendly and professional tone"""
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all conversation messages."""
        return self.messages
    
    def reset(self) -> None:
        """Reset conversation, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """Export conversation to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/conversations/conversation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)
        
        return filename