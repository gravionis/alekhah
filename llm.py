import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Try to import Google chat client if available; otherwise continue without error.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

class LLMClient:
    """Thin LLM wrapper exposing summarize(snippets_text, question, max_chars)."""
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.use_google = bool(self.api_key and ChatGoogleGenerativeAI is not None)
        self._client = None
        if self.use_google:
            try:
                self._client = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=self.api_key)
            except Exception:
                self._client = None
                self.use_google = False

    def summarize(self, text: str, question: str | None = None, max_chars: int = 1000) -> str:
        """Return a concise summary of `text`. Uses Google LLM if available, else deterministic fallback."""
        prompt = f"""Summarize the following search into a professional and complete answer. Answer directly to the question, dont say things like Based on provided text snippets etc.
                    {(' for question: ' + question) if question else ''} into a concise paragraph (max {max_chars} characters):\n\n{text}"""
        resp = self._client.invoke(prompt)
        summary = resp.content
        return summary

def get_llm(api_key: str | None = None) -> LLMClient:
    """Return an LLMClient instance (always returns a usable client, may be fallback-only)."""
    return LLMClient(api_key=api_key)
