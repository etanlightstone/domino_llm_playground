import requests
from langchain_community.llms import HuggingFacePipeline
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Optional, Any, Dict, Union
import torch
from pydantic import Field

class EtanHuggingFaceClient:
    def __init__(self):
        """Initialize the HuggingFace client."""
        pass
    
    def get_langchain_huggingface_model(self, model_name: str, temperature: float = 0.7, 
                                       max_length: int = 2048, device_map: str = "auto",
                                       load_in_4bit: bool = False):
        """Get a LangChain LLM from a local HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model
            temperature: Sampling temperature for generation
            max_length: Maximum length of generated text
            device_map: Device to run model on ('auto', 'cpu', 'cuda', 'mps', etc.)
            load_in_4bit: Whether to load model in 4-bit quantization
            
        Returns:
            A LangChain LLM wrapper for the HuggingFace model
        """
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization if requested
        model_kwargs = {"device_map": device_map}
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model with the appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # Create a LangChain HF Pipeline wrapper
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        
        return hf_pipeline
    
    def get_langchain_huggingface_chat_model(self, model_name: str, temperature: float = 0.7, 
                                            max_length: int = 2048, device_map: str = "auto",
                                            load_in_4bit: bool = False):
        """Get a LangChain chat model from a local HuggingFace model that supports invoke(messages).
        
        Args:
            model_name: Name of the HuggingFace model
            temperature: Sampling temperature for generation
            max_length: Maximum length of generated text
            device_map: Device to run model on ('auto', 'cpu', 'cuda', 'mps', etc.)
            load_in_4bit: Whether to load model in 4-bit quantization
            
        Returns:
            A chat model that can be used with the invoke(messages) interface
        """
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization if requested
        model_kwargs = {"device_map": device_map}
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model with the appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95
        )
        
        # Create a LangChain HF Pipeline wrapper
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        
        # Wrap in a custom chat model that can handle invoke(messages)
        return HuggingFaceChatModel(llm=hf_pipeline, tokenizer=tokenizer)

class HuggingFaceChatModel(BaseChatModel):
    """Custom chat model wrapper for HuggingFace models."""
    
    llm: HuggingFacePipeline = Field(exclude=False)
    tokenizer: Any = Field(exclude=False)
    
    def __init__(self, llm: HuggingFacePipeline, tokenizer):
        """Initialize the chat model wrapper.
        
        Args:
            llm: The underlying LangChain LLM
            tokenizer: The HuggingFace tokenizer
        """
        super().__init__(llm=llm, tokenizer=tokenizer)
        
    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        """Generate a response based on the given messages.
        
        Args:
            messages: List of LangChain message objects
            stop: Optional stop sequences
            run_manager: Optional run manager
            
        Returns:
            A ChatResult with the generated response
        """
        prompt = self._convert_messages_to_prompt(messages)
        
        # Call the underlying language model
        response = self.llm(prompt, stop=stop, **kwargs)
        
        # Create an AI message from the response
        message = AIMessage(content=response)
        
        # Create a ChatResult with the generated message
        generations = [ChatGeneration(message=message)]
        return ChatResult(generations=generations)
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a prompt string.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            A formatted prompt string
        """
        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and callable(self.tokenizer.apply_chat_template):
            # Convert LangChain messages to format expected by HF tokenizer
            messages_dict = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    messages_dict.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    messages_dict.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages_dict.append({"role": "assistant", "content": message.content})
            
            # Apply the chat template
            prompt = self.tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
            return prompt
        
        # Fallback to basic concatenation if no chat template
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"
        prompt += "AI: "
        return prompt
    
    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "huggingface-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"llm": self.llm, "tokenizer": str(self.tokenizer.__class__)}

def get_hf_qwen():
    client = EtanHuggingFaceClient()
    
    
    # # Example with standard LLM interface
    # llm = client.get_langchain_huggingface_model(
    #     #model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     model_name="Qwen/Qwen2.5-7B-Instruct",
    #     device_map="auto",  # Will use GPU if available
    #     load_in_4bit=True   # Use 4-bit quantization to reduce memory usage
    # )
    # response = llm.invoke("Tell me about artificial intelligence.")
    # print(f"LLM Response:\n{response}")
    
    # Example with chat interface
    chat_model = client.get_langchain_huggingface_chat_model(
        #model_name="mistralai/Mistral-7B-Instruct-v0.2", 
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        load_in_4bit=True
    )
    return chat_model
    
# Example usage
if __name__ == "__main__":
    chat_model = get_hf_qwen()
    from langchain_core.messages import HumanMessage, SystemMessage
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Tell me about artificial intelligence.")
    ]
    
    response = chat_model.invoke(messages)
    print(f"Chat Response:\n{response.content}")
    
