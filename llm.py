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
                print("here.................")
                # Lazy instantiate client; keep simple configuration
                self._client = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=self.api_key)
            except Exception:
                self._client = None
                self.use_google = False

    def summarize(self, text: str, question: str | None = None, max_chars: int = 1000) -> str:
        """Return a concise summary of `text`. Uses Google LLM if available, else deterministic fallback."""
        prompt = f"Summarize the following search snippets{(' for question: ' + question) if question else ''} into a concise paragraph (max {max_chars} characters):\n\n{text}"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",prompt)
        # Try Google LLM if configured
        if self.use_google and self._client is not None:
            try:
                resp = self._client.invoke(prompt)
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<", resp)
                # coerce to string safely
                if isinstance(resp, str):
                    summary = resp.strip()
                else:
                    summary = str(resp).strip()
            except Exception:
                summary = ""
        else:
            summary = ""

        # Deterministic fallback if LLM failed or not configured
        if not summary:
            if not text:
                summary = ""
            else:
                if len(text) <= max_chars:
                    summary = text
                else:
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    out = []
                    total = 0
                    for sent in sentences:
                        if total + len(sent) > max_chars:
                            break
                        out.append(sent)
                        total += len(sent) + 1
                    if out:
                        summary = " ".join(out)
                    else:
                        summary = text[:max_chars].rsplit(" ", 1)[0] + "..."
        # final safety trim
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
        return summary

def get_llm(api_key: str | None = None) -> LLMClient:
    """Return an LLMClient instance (always returns a usable client, may be fallback-only)."""
    return LLMClient(api_key=api_key)
