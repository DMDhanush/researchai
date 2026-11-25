"""
ResearchAI — LLM Provider Abstraction
Supports OpenAI and Ollama with auto-detection.
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Provider-agnostic LLM client supporting OpenAI and Ollama."""

    def __init__(self):
        self.provider = self._initialize_provider()
        self.openai_client = None
        self.ollama_base_url = None

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise RuntimeError("OpenAI library not installed. Install with: pip install openai")
        elif self.provider == "ollama":
            import requests
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Test connection
            try:
                requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            except Exception as e:
                raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_base_url}: {e}")

    def _initialize_provider(self) -> str:
        """Initialize and return the active LLM provider."""
        provider = os.getenv("LLM_PROVIDER", "auto").lower()

        if provider == "auto":
            # Try OpenAI first if API key exists
            if os.getenv("OPENAI_API_KEY"):
                return "openai"
            # Then try Ollama
            import requests
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                requests.get(f"{ollama_url}/api/tags", timeout=2)
                return "ollama"
            except:
                pass
            # Neither available
            raise RuntimeError(
                "Generative AI provider is not configured. "
                "Add an OpenAI API key or configure Ollama."
            )
        elif provider in ["openai", "ollama"]:
            return provider
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            prompt: User query
            context: Retrieved evidence/context from RAG
            system_prompt: System instruction for the LLM
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response length

        Returns:
            Generated text from the LLM
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        # Combine system prompt and context
        full_system = system_prompt
        if context:
            full_system += f"\n\nContext from research papers:\n{context}"

        if self.provider == "openai":
            return self._generate_openai(full_system, prompt, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._generate_ollama(full_system, prompt, temperature, max_tokens)

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI API."""
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Ollama local API."""
        import requests

        model = os.getenv("OLLAMA_MODEL", "llama3.2")

        payload = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def get_provider(self) -> str:
        """Return the active provider name."""
        return self.provider


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
