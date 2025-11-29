"""
Customer Support Chatbot Application with Memory
A production-ready chatbot with persistent user memory named Alexa.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ChatbotConfig:
    """Configuration management for the chatbot."""
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
        self.timeout = 30
        self.memory_file = "data/user_memory.json"
        
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Please create a .env file with your API key.")
            return False
        return True


class UserMemory:
    """Manages persistent user information."""
    
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """Load user memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_memory(self) -> None:
        """Save user memory to file."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
    
    def get_user_name(self) -> Optional[str]:
        """Get stored user name."""
        return self.memory.get('user_name')
    
    def set_user_name(self, name: str) -> None:
        """Store user name."""
        self.memory['user_name'] = name
        self.memory['last_updated'] = datetime.now().isoformat()
        self._save_memory()
    
    def get_memory_summary(self) -> str:
        """Get a summary of stored memory for the system prompt."""
        if self.memory.get('user_name'):
            return f"The user's name is {self.memory['user_name']}."
        return ""
    
    def clear_memory(self) -> None:
        """Clear all stored memory."""
        self.memory = {}
        self._save_memory()


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, user_memory: UserMemory, system_prompt: Optional[str] = None):
        self.messages: List[Dict[str, str]] = []
        self.user_memory = user_memory
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.messages.append({"role": "system", "content": self.system_prompt})
        
    def _default_system_prompt(self) -> str:
        """Returns default system prompt for customer support."""
        memory_info = self.user_memory.get_memory_summary()
        base_prompt = """You are Alexa, a helpful and professional customer support assistant. 
        Your role is to:
        - Answer customer questions clearly and concisely
        - Be polite, patient, and empathetic
        - Provide accurate information
        - Remember and use the user's name when appropriate
        - Maintain a friendly and professional tone
        
        IMPORTANT: 
        - Your name is Alexa
        - When asked what model you are, say you are "Alexa, powered by GPT-4o Mini"
        - If the user tells you their name, acknowledge it warmly and remember it
        - Always use the user's name naturally in conversation when you know it"""
        
        if memory_info:
            base_prompt += f"\n\nREMEMBERED INFORMATION:\n{memory_info}"
        
        return base_prompt
    
    def add_user_message(self, content: str) -> None:
        """Add a user message and check for name mentions."""
        self.messages.append({"role": "user", "content": content})
        self._extract_user_name(content)
    
    def _extract_user_name(self, message: str) -> None:
        """Try to extract user name from message."""
        lower_msg = message.lower()
        
        # Common patterns for name introduction
        patterns = [
            "my name is ",
            "i'm ",
            "i am ",
            "call me ",
            "this is "
        ]
        
        for pattern in patterns:
            if pattern in lower_msg:
                start = lower_msg.index(pattern) + len(pattern)
                # Get the next word (name)
                words = message[start:].split()
                if words:
                    name = words[0].strip('.,!?').capitalize()
                    if len(name) > 1 and name.isalpha():
                        self.user_memory.set_user_name(name)
                        print(f"\n[Memory Updated: User name saved as '{name}']\n")
                        break
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all conversation messages."""
        return self.messages
    
    def reset(self) -> None:
        """Reset conversation, keeping memory and system prompt."""
        self.system_prompt = self._default_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """Export conversation to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/conversations/conversation_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)
        
        return filename


class OpenAIChatbot:
    """Handles communication with OpenAI API."""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.user_memory = UserMemory(config.memory_file)
        self.conversation = ConversationManager(self.user_memory)
    
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
        print("Conversation has been reset. (User memory retained)")
    
    def export_conversation(self, filename: Optional[str] = None) -> None:
        """Export conversation history."""
        try:
            filepath = self.conversation.export_conversation(filename)
            print(f"Conversation exported to: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to export: {e}")
    
    def forget_me(self) -> None:
        """Clear all user memory."""
        self.user_memory.clear_memory()
        self.conversation.reset()
        print("All user memory has been cleared.")
    
    def show_memory(self) -> None:
        """Display stored user information."""
        name = self.user_memory.get_user_name()
        if name:
            print(f"\n📝 Stored Information:")
            print(f"   Your name: {name}")
        else:
            print("\n📝 No information stored yet.")
        print()


def print_welcome_message():
    """Print welcome message and instructions."""
    print("\n" + "="*60)
    print("       ALEXA - CUSTOMER SUPPORT CHATBOT")
    print("="*60)
    print("\nHello! I'm Alexa, your customer support assistant.")
    print("I'll remember your name across sessions!")
    print("\nCommands:")
    print("  - Type your question to chat")
    print("  - '/reset' to start a new conversation")
    print("  - '/export' to save conversation history")
    print("  - '/memory' to see what I remember about you")
    print("  - '/forget' to clear all memory")
    print("  - '/quit' or '/exit' to end the session")
    print("="*60 + "\n")


def main():
    """Main application loop."""
    config = ChatbotConfig()
    
    if not config.validate():
        sys.exit(1)
    
    chatbot = OpenAIChatbot(config)
    print_welcome_message()
    
    # Greet user by name if remembered
    user_name = chatbot.user_memory.get_user_name()
    if user_name:
        print(f"👋 Welcome back, {user_name}!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit']:
                print("\nThank you for chatting with Alexa. Goodbye!")
                break
            
            elif user_input.lower() == '/reset':
                chatbot.reset_conversation()
                continue
            
            elif user_input.lower() == '/export':
                chatbot.export_conversation()
                continue
            
            elif user_input.lower() == '/memory':
                chatbot.show_memory()
                continue
            
            elif user_input.lower() == '/forget':
                confirm = input("Are you sure you want to clear all memory? (yes/no): ")
                if confirm.lower() == 'yes':
                    chatbot.forget_me()
                continue
            
            response = chatbot.send_message(user_input)
            
            if response:
                print(f"\nAlexa: {response}\n")
            else:
                print("Failed to get response. Please try again.\n")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting...")
            break
        
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


if __name__ == "__main__":
    main()