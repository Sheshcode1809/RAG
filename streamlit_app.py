import streamlit as st
import os
from dotenv import load_dotenv
from chat import chat_stream

load_dotenv()

st.set_page_config(
    page_title="Textbook RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Textbook RAG Assistant")
st.caption("Ask questions strictly based on the textbook. Powered by Hybrid Retrieval (BM25 + Chroma) and Groq.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 Sources & Citations"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}** — Page {src['page']}")

# Chat input
if user_input := st.chat_input("Ask a question about your textbook..."):
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare history for chat function
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks and preparing context..."):
            result = chat_stream(user_input, history=history)

        answer_stream = result.get("answer_stream")
        sources = result.get("sources", [])

        if answer_stream is not None:
            # Stream tokens word-by-word
            full_response = st.write_stream(answer_stream)
        else:
            full_response = result.get("answer", "No response generated.")
            st.markdown(full_response)

        # Show citations
        if sources:
            with st.expander("📖 Sources & Citations"):
                for src in sources:
                    st.markdown(f"- **{src['source']}** — Page {src['page']}")

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
