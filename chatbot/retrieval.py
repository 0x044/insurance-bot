from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import re

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """
    Chunk text with overlap and attempt to preserve paragraph boundaries
    """
    # Clean text and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Try to split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, store current chunk and start new one
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # For long paragraphs, split by sentences
            if len(paragraph) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                    else:
                        current_chunk += sentence + " "
            else:
                current_chunk = paragraph + " "
        else:
            current_chunk += paragraph + " "
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlapping chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            overlapped_chunks.append(chunks[i])
            
            # Create overlap chunk using end of current and start of next
            words_current = chunks[i].split()
            words_next = chunks[i+1].split()
            
            if len(words_current) > chunk_overlap and len(words_next) > chunk_overlap:
                overlap_text = " ".join(words_current[-chunk_overlap:] + words_next[:chunk_overlap])
                overlapped_chunks.append(overlap_text)
        
        # Add the last chunk
        overlapped_chunks.append(chunks[-1])
        chunks = overlapped_chunks
    
    return chunks

def create_embeddings(chunks):
    """Create embeddings with progress tracking"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process in batches for large documents
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        print(f"Encoded {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    return model, np.array(embeddings)

def save_index(embeddings, chunks, path="faiss_index"):
    """Save index with additional metadata"""
    os.makedirs(path, exist_ok=True)
    
    # Use IVFFlat index for faster retrieval with large document collections
    dimension = embeddings.shape[1]
    
    # For smaller collections, use simple IndexFlatL2
    if len(chunks) < 1000:
        index = faiss.IndexFlatL2(dimension)
    else:
        # For larger collections, use IVFFlat for better performance
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = min(100, int(len(chunks) / 10))  # Number of clusters
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, f"{path}/index.faiss")
    
    # Save chunks with metadata
    chunk_data = {
        "chunks": chunks,
        "count": len(chunks),
        "created_at": os.path.getmtime(f"{path}/index.faiss") if os.path.exists(f"{path}/index.faiss") else None,
    }
    
    with open(f"{path}/chunks.pkl", "wb") as f:
        pickle.dump(chunk_data, f)

def load_index(path="faiss_index"):
    """Load index with error handling"""
    try:
        index = faiss.read_index(f"{path}/index.faiss")
        
        with open(f"{path}/chunks.pkl", "rb") as f:
            chunk_data = pickle.load(f)
            
        # Handle both formats (old and new)
        if isinstance(chunk_data, dict):
            chunks = chunk_data["chunks"]
        else:
            chunks = chunk_data
            
        return index, chunks
        
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        # Return empty index and chunks if load fails
        dimension = 384  # Default for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(dimension)
        return index, []    