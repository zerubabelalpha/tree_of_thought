import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environmental variables
load_dotenv()

class LLMClient:
    
    def __init__(self, api_key: str = None, model: str = None):
        # Fallback to loading from environment variables if not passed directly
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set"
            )
            
        # Instantiate standard OpenAI client pointing to OpenRouter API base
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def complete(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 3000) -> str:
        #Sends a query to OpenRouter via the OpenAI SDK and returns the text response.
        
        messages = []   
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    raise Exception("Received empty completion content from LLM choice.")
            else:
                raise Exception("No choices returned from OpenRouter response completions.")
                
        except Exception as e:
            error_msg = f"OpenRouter API request failed via OpenAI client: {e}"
            raise Exception(error_msg)
