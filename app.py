import streamlit as st
from chatbot import handle_query
from chatbot.knowledge_base import extract_text_from_pdf
from chatbot.retrieval import chunk_text, create_embeddings, save_index, load_index
import os

st.set_page_config(page_title="Insurance Chatbot", layout="wide")
st.title("🛡️ Insurance Policy Information Chatbot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize knowledge base
if 'index' not in st.session_state or 'chunks' not in st.session_state:
    try:
        if not os.path.exists("faiss_index/index.faiss"):
            with st.spinner("Initializing knowledge base..."):
                text = extract_text_from_pdf("data/policy_documents.pdf")
                # Use smaller chunk size and overlap for better context
                chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
                model, embeddings = create_embeddings(chunks)
                save_index(embeddings, chunks)
                st.success("Knowledge base ready!")
        index, chunks = load_index()
        st.session_state.index = index
        st.session_state.chunks = chunks
    except Exception as e:
        st.error(f"Error initializing knowledge base: {str(e)}")
        st.stop()

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        confidence = chat.get("confidence", 0)
        with st.chat_message("assistant"):
            st.write(chat["content"])
            if confidence < 0.5:
                st.warning(f"Low confidence response ({confidence:.2f})")
            elif confidence < 0.7:
                st.info(f"Medium confidence response ({confidence:.2f})")
            else:
                st.success(f"High confidence response ({confidence:.2f})")

# Get user input
user_input = st.chat_input("Ask a question about our insurance policies:")

if user_input:
    # Display user message
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get previous conversation for context
    conversation_context = ""
    if len(st.session_state.chat_history) > 1:
        # Get last 3 exchanges for context
        recent_messages = st.session_state.chat_history[-6:]
        conversation_context = "\n".join([f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" 
                                         for msg in recent_messages[:-1]])
    
    # Get response
    with st.spinner("Thinking..."):
        try:
            response, confidence, sources = handle_query(
                user_input, 
                st.session_state.index, 
                st.session_state.chunks,
                conversation_context
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
                
                # Display confidence indicator
                if confidence < 0.5:
                    st.warning(f"Low confidence response ({confidence:.2f})")
                elif confidence < 0.7:
                    st.info(f"Medium confidence response ({confidence:.2f})")
                else:
                    st.success(f"High confidence response ({confidence:.2f})")
                
                # Display sources if available
                if sources:
                    with st.expander("Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**\n```\n{source}\n```")
            
            # Add to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "confidence": confidence
            })
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I'm sorry, I encountered an error processing your question. Please try again.",
                "confidence": 0
            })

# Add sidebar with information
with st.sidebar:
    st.subheader("About this chatbot")
    st.write("""
    This chatbot provides information about insurance policies based on the uploaded documents.
    It uses AI to understand your questions and find relevant information.
    
    If the confidence score is low, the answer might not be accurate.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()