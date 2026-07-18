# pyrefly: ignore [missing-import]
import chainlit as cl
from chat import chat_stream_async

@cl.on_chat_start
async def start():
    # Initialize conversation history
    cl.user_session.set("history", [])

    await cl.Message(
        content=(
            "# 📚 Textbook RAG Assistant\n\n"
            "Ask me questions about the uploaded textbooks.\n\n"
            "I'll answer only from the textbook content and provide page citations."
        )
    ).send()
    
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    # Use cl.Step to show loading spinner for the RAG search and context preparation phase
    async with cl.Step(name="Searching textbooks and preparing context...") as step:
        result = await chat_stream_async(
            question=message.content,
            history=history
        )
        step.input = message.content
        
        # Display feedback in the step description
        num_sources = len(result.get("sources", []))
        if result.get("answer_stream") is not None:
            step.output = f"Retrieved context from {num_sources} matching pages. Ready to generate response."
        else:
            step.output = "No matching context found."

    answer_stream = result.get("answer_stream")
    sources = result.get("sources", [])

    # Send an empty message to start the UI bubble
    msg = cl.Message(content="")
    full_answer = ""

    if answer_stream is not None:
        await msg.send()
        # Stream the tokens word-by-word in real-time
        async for token in answer_stream:
            await msg.stream_token(token)
            full_answer += token
        await msg.update()
    else:
        # Static fallback answer (e.g. no matching textbooks found)
        full_answer = result.get("answer", "No response generated.")
        msg.content = full_answer
        await msg.send()

    # Append page-level sources/citations if available
    if sources:
        citation_text = "\n\n---\n### Sources\n"
        for src in sources:
            citation_text += f"- **{src['source']}** — Page {src['page']}\n"
        
        msg.content = full_answer + citation_text
        await msg.update()

    # Update conversation history with the new exchange
    history.append(
        {
            "role": "user",
            "content": message.content
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": full_answer
        }
    )
    cl.user_session.set("history", history)