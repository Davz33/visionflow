"""
Configuration for LLM providers used in subjective video evaluation.

This module allows easy switching between different LLM providers:
- LLaVA (local deployment, open-source)
- Gemini Pro Vision (API-based, Google)
- GPT-4V (API-based, OpenAI)
"""

from enum import Enum
from typing import Optional
from pathlib import Path
import os

class LLMProvider(Enum):
    """Available LLM providers for subjective evaluation."""
    LLAVA = "llava"
    GEMINI = "gemini"
    GPT4V = "gpt4v"

class LLMConfig:
    """Configuration for LLM providers."""
    
    # Default provider
    DEFAULT_PROVIDER = LLMProvider.LLAVA
    
    # LLaVA configuration
    LLAVA_MODEL_PATH = os.getenv("LLAVA_MODEL_PATH", "liuhaotian/llava-v1.5-13b")
    LLAVA_DEVICE = os.getenv("LLAVA_DEVICE", "auto")  # auto, cuda, cpu
    LLAVA_MAX_TOKENS = int(os.getenv("LLAVA_MAX_TOKENS", "512"))
    LLAVA_TEMPERATURE = float(os.getenv("LLAVA_TEMPERATURE", "0.7"))
    
    # Gemini configuration
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro-vision")
    GEMINI_PROJECT = os.getenv("GEMINI_PROJECT", "")
    GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
    
    # GPT-4V configuration
    GPT4V_MODEL = os.getenv("GPT4V_MODEL", "gpt-4-vision-preview")
    GPT4V_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GPT4V_MAX_TOKENS = int(os.getenv("GPT4V_MAX_TOKENS", "1000"))
    GPT4V_TEMPERATURE = float(os.getenv("GPT4V_TEMPERATURE", "0.3"))
    
    @classmethod
    def get_provider(cls) -> LLMProvider:
        """Get the configured LLM provider."""
        provider_str = os.getenv("LLM_PROVIDER", cls.DEFAULT_PROVIDER.value)
        try:
            return LLMProvider(provider_str)
        except ValueError:
            print(f"Invalid LLM_PROVIDER: {provider_str}. Using default: {cls.DEFAULT_PROVIDER.value}")
            return cls.DEFAULT_PROVIDER
    
    @classmethod
    def is_llava_enabled(cls) -> bool:
        """Check if LLaVA is the selected provider."""
        return cls.get_provider() == LLMProvider.LLAVA
    
    @classmethod
    def is_gemini_enabled(cls) -> bool:
        """Check if Gemini is the selected provider."""
        return cls.get_provider() == LLMProvider.GEMINI
    
    @classmethod
    def is_gpt4v_enabled(cls) -> bool:
        """Check if GPT-4V is the selected provider."""
        return cls.get_provider() == LLMProvider.GPT4V
    
    @classmethod
    def get_llava_config(cls) -> dict:
        """Get LLaVA configuration."""
        return {
            "model_path": cls.LLAVA_MODEL_PATH,
            "device": cls.LLAVA_DEVICE,
            "max_tokens": cls.LLAVA_MAX_TOKENS,
            "temperature": cls.LLAVA_TEMPERATURE
        }
    
    @classmethod
    def get_gemini_config(cls) -> dict:
        """Get Gemini configuration."""
        return {
            "model_name": cls.GEMINI_MODEL,
            "project": cls.GEMINI_PROJECT,
            "location": cls.GEMINI_LOCATION,
            "max_output_tokens": cls.GEMINI_MAX_TOKENS,
            "temperature": cls.GEMINI_TEMPERATURE
        }
    
    @classmethod
    def get_gpt4v_config(cls) -> dict:
        """Get GPT-4V configuration."""
        return {
            "model": cls.GPT4V_MODEL,
            "api_key": cls.GPT4V_API_KEY,
            "max_tokens": cls.GPT4V_MAX_TOKENS,
            "temperature": cls.GPT4V_TEMPERATURE
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate the current configuration."""
        provider = cls.get_provider()
        
        if provider == LLMProvider.LLAVA:
            # LLaVA doesn't need API keys, just model path
            return bool(cls.LLAVA_MODEL_PATH)
        
        elif provider == LLMProvider.GEMINI:
            # Gemini needs project and location
            return bool(cls.GEMINI_PROJECT and cls.GEMINI_LOCATION)
        
        elif provider == LLMProvider.GPT4V:
            # GPT-4V needs API key
            return bool(cls.GPT4V_API_KEY)
        
        return False
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        provider = cls.get_provider()
        print(f"Current LLM Provider: {provider.value}")
        print(f"Configuration Valid: {cls.validate_config()}")
        
        if provider == LLMProvider.LLAVA:
            print("LLaVA Config:")
            for key, value in cls.get_llava_config().items():
                print(f"  {key}: {value}")
        
        elif provider == LLMProvider.GEMINI:
            print("Gemini Config:")
            for key, value in cls.get_gemini_config().items():
                if key == "project":
                    print(f"  {key}: {'***' if value else 'NOT SET'}")
                else:
                    print(f"  {key}: {value}")
        
        elif provider == LLMProvider.GPT4V:
            print("GPT-4V Config:")
            for key, value in cls.get_gpt4v_config().items():
                if key == "api_key":
                    print(f"  {key}: {'***' if value else 'NOT SET'}")
                else:
                    print(f"  {key}: {value}")


# Environment variable examples for easy setup
ENV_EXAMPLES = """
# LLaVA Configuration (Default)
LLM_PROVIDER=llava
LLAVA_MODEL_PATH=liuhaotian/llava-v1.5-13b
LLAVA_DEVICE=auto
LLAVA_MAX_TOKENS=512
LLAVA_TEMPERATURE=0.7

# Gemini Configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-pro-vision
GEMINI_PROJECT=your-gcp-project-id
GEMINI_LOCATION=us-central1
GEMINI_MAX_TOKENS=1024
GEMINI_TEMPERATURE=0.3

# GPT-4V Configuration
LLM_PROVIDER=gpt4v
GPT4V_MODEL=gpt-4-vision-preview
OPENAI_API_KEY=your-openai-api-key
GPT4V_MAX_TOKENS=1000
GPT4V_TEMPERATURE=0.3
"""
