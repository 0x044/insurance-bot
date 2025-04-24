from chatbot.llm import generate_response
from chatbot.retrieval import load_index
from sentence_transformers import SentenceTransformer
import numpy as np
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_similar_context(query, index, chunks, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    
    # Get the top chunks and their similarity scores
    context_chunks = []
    scores = []
    for i, idx in enumerate(I[0]):
        if idx < len(chunks):  # Ensure index is valid
            context_chunks.append(chunks[idx])
            scores.append(D[0][i])
    
    # Calculate overall confidence based on top similarity score
    # Lower distance means higher similarity
    if not scores:
        return 0.0, "", []
    
    # Convert distance to similarity score (0-1 range)
    # Closer to 0 distance means higher similarity
    max_distance = 2.0  # Maximum possible L2 distance for normalized vectors is √2
    confidence = max(0.0, 1.0 - (scores[0] / max_distance))
    
    return confidence, "\n".join(context_chunks), context_chunks[:3]  # Return top 3 chunks as sources

def handle_query(user_input, index, chunks, conversation_context=""):
    # Check for empty or very short queries
    if not user_input or len(user_input.strip()) < 3:
        return "Please ask a more detailed question about our insurance policies.", 0.0, []
    
    # Get relevant context
    confidence, context, sources = get_similar_context(user_input, index, chunks)
    
    # Create a detailed prompt with clear instructions to reduce hallucination
    prompt = f"""You are an insurance policy assistant. Your task is to answer the user's question based ONLY on the information provided in the context. 
If the context doesn't contain relevant information to answer the question, admit that you don't know rather than making up an answer.

Previous conversation:
{conversation_context}

Context information from insurance policy documents:
{context}

User's question: {user_input}

Remember:
1. Only use facts stated in the context.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question accurately."
3. Don't make up policy details not found in the context.
4. Be clear and concise.

Your answer:"""

    # Adjust max tokens based on the length of the user query
    max_tokens = min(500, max(150, len(user_input) * 3))
    
    # Generate response with confidence level
    response = generate_response(prompt, confidence, max_tokens)
    
    return response, confidence, sources