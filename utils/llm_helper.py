"""
Shared Groq LLM helper — drop-in replacement for Ollama.
Falls back to Ollama if Groq key not available (local dev).
"""
import os

def get_llm(temperature=0.1, max_tokens=1024):
    """Return Groq LLM if API key available, else Ollama."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Groq init failed: {e}")
    
    # Fallback to Ollama for local dev
    try:
        from langchain_community.llms import Ollama
        return Ollama(model="llama3.2", temperature=temperature)
    except Exception as e:
        print(f"Ollama fallback failed: {e}")
        return None

def call_groq(prompt, system="You are an IBM delivery consultant AI assistant.", temperature=0.1):
    """Direct Groq API call — returns string response."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Groq error: {e}"
    
    # Fallback to Ollama
    try:
        import requests
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": f"{system}\n\n{prompt}", "stream": False},
            timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    
    return "LLM not available. Please set GROQ_API_KEY in your .env file."
