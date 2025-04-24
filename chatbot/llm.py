from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Use environment variable for model selection if available
model_name = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Add caching to improve performance
class CachedLLM:
    def __init__(self, model_name):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Response templates based on confidence
        self.templates = {
            "high": """Based on the insurance policy information, I can confidently answer your question:

{response}""",
            "medium": """Based on the available information, here's what I found about your question:

{response}

Please note that you may want to verify this information with your policy documents or an insurance representative.""",
            "low": """I'm not entirely certain, but based on the limited information I have:

{response}

I recommend checking your specific policy documents or contacting customer service for confirmation.""",
            "no_info": """I don't have enough information in the policy documents to answer this question accurately. 

Please contact customer service at the number on your insurance card for specific information about this topic."""
        }

    def generate_response(self, prompt, max_new_tokens=300, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with some randomness for more natural responses
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return response
    
    def format_response(self, raw_response, confidence):
        """Format response based on confidence level"""
        if confidence < 0.3:
            template = self.templates["no_info"]
        elif confidence < 0.5:
            template = self.templates["low"]
        elif confidence < 0.7:
            template = self.templates["medium"]
        else:
            template = self.templates["high"]
            
        return template.format(response=raw_response)

# Initialize model
llm = CachedLLM(model_name)

def generate_response(prompt, confidence=0.7, max_tokens=300):
    """Generate response with appropriate template based on confidence"""
    raw_response = llm.generate_response(prompt, max_new_tokens=max_tokens)
    return llm.format_response(raw_response, confidence)