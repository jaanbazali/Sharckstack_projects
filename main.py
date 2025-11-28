"""Main application entry point."""

import sys
from src.config import ChatbotConfig
from src.chatbot import OpenAIChatbot
from src.utils import print_welcome_message


def main():
    """Main application loop."""
    config = ChatbotConfig()
    
    if not config.validate():
        sys.exit(1)
    
    chatbot = OpenAIChatbot(config)
    print_welcome_message()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit']:
                print("\nThank you for using our support chatbot. Goodbye!")
                break
            
            elif user_input.lower() == '/reset':
                chatbot.reset_conversation()
                continue
            
            elif user_input.lower() == '/export':
                chatbot.export_conversation()
                continue
            
            response = chatbot.send_message(user_input)
            
            if response:
                print(f"\nAssistant: {response}\n")
            else:
                print("Failed to get response. Please try again.\n")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting...")
            break
        
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


if __name__ == "__main__":
    main()