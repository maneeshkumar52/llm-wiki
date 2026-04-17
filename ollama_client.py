"""Small wrapper around the Ollama HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests


class OllamaClient:
    """Client for interacting with local or remote Ollama instances."""

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """Return True when Ollama responds to a lightweight request."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return True

    def list_models(self) -> list[str]:
        """Return locally available Ollama model names."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [str(model.get("name")) for model in models if model.get("name")]

    def generate_answer(
        self,
        model: str,
        question: str,
        context_blocks: list[str],
        system_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate an answer grounded in retrieved context."""
        context = "\n\n".join(
            f"Context {index + 1}:\n{chunk}" for index, chunk in enumerate(context_blocks)
        )
        prompt = (
            "Answer the user's question using only the supplied context when possible. "
            "If the answer is not present in the context, say that clearly and suggest what document might help.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context}"
        )
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "system": system_prompt,
            "prompt": prompt,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

    def stream_answer(
        self,
        model: str,
        question: str,
        context_blocks: list[str],
        system_prompt: str,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """Stream a grounded answer chunk by chunk from Ollama."""
        context = "\n\n".join(
            f"Context {index + 1}:\n{chunk}" for index, chunk in enumerate(context_blocks)
        )
        prompt = (
            "Answer the user's question using only the supplied context when possible. "
            "If the answer is not present in the context, say that clearly and suggest what document might help.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context}"
        )
        payload = {
            "model": model,
            "stream": True,
            "options": {"temperature": temperature},
            "system": system_prompt,
            "prompt": prompt,
        }
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                payload = json.loads(raw_line)
                content = payload.get("response")
                if content:
                    yield str(content)
                if payload.get("done"):
                    break

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Create embeddings using an Ollama embedding model."""
        embeddings: list[list[float]] = []
        for text in texts:
            payload: dict[str, Any] = {"model": model, "input": text}
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embeddings") or []
            if not vector:
                raise ValueError("Ollama did not return an embedding vector")
            embeddings.append(vector[0])
        return embeddings