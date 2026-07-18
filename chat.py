from retrieve import retrieve
from groq import Groq, AsyncGroq
from dotenv import load_dotenv
import os

load_dotenv()

SIMILARITY_THRESHOLD = 0.45

# Clients
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

async_client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Custom sliding window memory to prevent context window overflow
class ConversationBufferWindowMemory:
    def __init__(self, k: int = 4):
        self.k = k

    def get_history_window(self, history):
        if not history:
            return []
        # Each exchange has 2 messages (user, assistant)
        # So last k exchanges = last k * 2 messages
        return history[-(self.k * 2):]

def build_prompt(question, retrieved_chunks, history=None):
    """
    Creates the prompt given to the LLM.
    """
    context = ""
    for chunk in retrieved_chunks:
        context += (
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )

    # Apply ConversationBufferWindowMemory (k=4 exchanges)
    memory = ConversationBufferWindowMemory(k=4)
    history_window = memory.get_history_window(history)

    conversation = ""
    for msg in history_window:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role_label}: {msg['content']}\n"

    prompt = f"""You are a helpful textbook assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT use outside knowledge.
- If the answer is not present, say:
  "The requested information is not available in the provided textbooks."
- Be concise and accurate.

Conversation History:
{conversation}

Context:
{context}

Question: {question}

Answer:
"""
    return prompt

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content

def ask_llm_stream(prompt):
    """Synchronous token streaming (used for CLI/tests)"""
    response_stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        stream=True
    )
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def ask_llm_stream_async(prompt):
    """Asynchronous token streaming (used for Chainlit)"""
    response_stream = await async_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        stream=True
    )
    async for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def chat_stream(question, history=None):
    """Sync chat stream interface"""
    retrieved = retrieve(question)

    if len(retrieved) == 0:
        return {
            "answer_stream": None,
            "answer": "No relevant information found.",
            "sources": []
        }

    best_score = retrieved[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer_stream": None,
            "answer": "The requested information is not available in the provided textbooks.",
            "sources": []
        }

    prompt = build_prompt(
        question,
        retrieved,
        history
    )

    answer_generator = ask_llm_stream(prompt)

    citations = []
    seen = set()

    for chunk in retrieved:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            citations.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            )
            seen.add(key)

    return {
        "answer_stream": answer_generator,
        "sources": citations
    }

async def chat_stream_async(question, history=None):
    """Async chat stream interface for Chainlit"""
    retrieved = retrieve(question)

    if len(retrieved) == 0:
        return {
            "answer_stream": None,
            "answer": "No relevant information found.",
            "sources": []
        }

    best_score = retrieved[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer_stream": None,
            "answer": "The requested information is not available in the provided textbooks.",
            "sources": []
        }

    prompt = build_prompt(
        question,
        retrieved,
        history
    )

    answer_generator = ask_llm_stream_async(prompt)

    citations = []
    seen = set()

    for chunk in retrieved:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            citations.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            )
            seen.add(key)

    return {
        "answer_stream": answer_generator,
        "sources": citations
    }

def chat(question, history=None):
    """Synchronous chat function, fully backwards compatible"""
    result = chat_stream(question, history)
    if result.get("answer_stream") is not None:
        answer = "".join(list(result["answer_stream"]))
        return {
            "answer": answer,
            "sources": result["sources"]
        }
    else:
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

if __name__ == "__main__":
    history = []
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            break

        result = chat(question, history)

        print("\nAnswer\n")
        print(result["answer"])

        print("\nSources")
        for src in result["sources"]:
            print(f"- {src['source']} (Page {src['page']})")

        history.append(
            {
                "role": "user",
                "content": question
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )