import requests
#from langchain.llms import Ollama
#from langchain_community.llms import ollama
#from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOllama

class EtanOllamaClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def get_langchain_ollama_model(self, model_name, callback_manager=None):
        return ChatOllama(
            base_url=self.api_url,
            model=model_name,
            callback_manager=callback_manager
        )
    #def get_langchain_ollama_model(self, model_name):
    def list_models(self):
        """Get a list of available models."""
        url = f"{self.api_url}/v1/models"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            models = response.json()
            model_ids = [model['id'] for model in models['data']]
            return model_ids
        else:
            raise Exception(f"Failed to get models: {response.text}")

    def chat(self, model_name, message):
        """Send a chat request and receive a response."""
        url = f"{self.api_url}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Failed to chat: {response.text}")

# Example usage
if __name__ == "__main__":
    # Replace with your Ollama API URL
    api_url = "http://localhost:11434"

    client = EtanOllamaClient(api_url)

    try:
        # List available models
        models = client.list_models()
        print("Available models:", models)
        model_name= models[0]

        # Chat with a model
        #model_name = "qwen2.5-coder:14b-instruct-q6_K"  # Replace with your desired model name
        message = "Hello, how are you?"
        response = client.chat(model_name, message)
        print(f"Response from {model_name}:\n\n {response}")

    except Exception as e:
        print("Error:", e)