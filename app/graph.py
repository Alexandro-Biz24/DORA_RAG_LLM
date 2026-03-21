from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from app.config import CHAT_MODEL, TEMPERATURE
from app.openai_client import client
from app.retrieval import retrieve, build_context_payload


SYSTEM_PROMPT = """
You are a legal research assistant.

Rules:
- Answer only from the provided context.
- Do not invent legal facts.
- If the answer is not clearly supported by the context, say so explicitly.
- Cite the relevant chunk_id(s) in your answer when making legal claims.
- Prefer precise, structured, neutral answers.
- If the user asks for a summary, keep legal precision.
- If the user asks for a comparison, separate the compared points clearly.
""".strip()


class GraphState(TypedDict, total=False):
    question: str
    chat_history: List[Dict]

    rewritten_query: str

    chunks: List[Dict]
    used_chunks: List[Dict]
    chunk_ids: List[str]
    context: str

    answer: str
    sources: List[Dict]


def format_chat_history(chat_history: Optional[List[Dict]]) -> str:
    if not chat_history:
        return ""

    parts = []
    for msg in chat_history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")

    return "\n".join(parts).strip()


def summarize_chat_history_for_retrieval(
    chat_history: Optional[List[Dict]],
    max_messages: int = 6,
    max_chars: int = 1500
) -> str:
    if not chat_history:
        return ""

    recent_messages = chat_history[-max_messages:]
    parts = []

    for msg in recent_messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")

    history_text = "\n".join(parts).strip()

    if len(history_text) > max_chars:
        history_text = history_text[-max_chars:]

    return history_text


def rewrite_query_with_history(
    question: str,
    chat_history: Optional[List[Dict]] = None
) -> str:
    history_text = summarize_chat_history_for_retrieval(chat_history)

    if not history_text:
        return question

    prompt = f"""
You are a legal search assistant.

Rewrite the user's last question into a fully explicit legal retrieval query.

Rules:
- Resolve references like "it", "this", "that", "the previous article", "the previous regulation"
- Include the relevant legal document if implied by the conversation
- Include article / chapter / section / title references if implied or explicitly mentioned
- Keep the rewritten query concise but precise
- Do NOT answer the user's question
- Do NOT add commentary
- Output ONLY the rewritten query

Conversation history:
{history_text}

Last user question:
{question}

Rewritten query:
""".strip()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query if rewritten_query else question


def build_user_prompt(
    question: str,
    context: str,
    chat_history: Optional[List[Dict]] = None
) -> str:
    history_text = format_chat_history(chat_history)

    prompt_parts = []

    if history_text:
        prompt_parts.append("Conversation history:")
        prompt_parts.append(history_text)
        prompt_parts.append("")

    prompt_parts.append("Retrieved legal context:")
    prompt_parts.append(context)
    prompt_parts.append("")
    prompt_parts.append("User question:")
    prompt_parts.append(question)
    prompt_parts.append("")
    prompt_parts.append(
        "Instructions: answer only from the retrieved legal context and cite chunk_id(s) for the statements you make."
    )

    return "\n".join(prompt_parts).strip()


def extract_sources_from_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    sources = []

    for ch in chunks:
        key = ch.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id"),
            "doc_title": ch.get("doc_title"),
            "doc_date": ch.get("doc_date"),
            "doc_subtitle": ch.get("doc_subtitle"),
            "title_nb": ch.get("title_nb"),
            "title_name": ch.get("title_name"),
            "chapter_nb": ch.get("chapter_nb"),
            "chapter_name": ch.get("chapter_name"),
            "section_nb": ch.get("section_nb"),
            "section_name": ch.get("section_name"),
            "article_nb": ch.get("article_nb"),
            "article_name": ch.get("article_name"),
            "block_type": ch.get("block_type"),
        })

    return sources


# =========================================================
# NODES
# =========================================================

def rewrite_query_node(state: GraphState) -> GraphState:
    question = state["question"]
    chat_history = state.get("chat_history", [])

    rewritten_query = rewrite_query_with_history(
        question=question,
        chat_history=chat_history
    )

    return {
        "rewritten_query": rewritten_query
    }


def retrieve_node(state: GraphState) -> GraphState:
    rewritten_query = state["rewritten_query"]

    chunks = retrieve(rewritten_query)

    return {
        "chunks": chunks
    }


def build_context_node(state: GraphState) -> GraphState:
    chunks = state["chunks"]

    payload = build_context_payload(chunks)

    return {
        "context": payload["context"],
        "chunk_ids": payload["chunk_ids"],
        "used_chunks": payload["used_chunks"],
        "sources": extract_sources_from_chunks(payload["used_chunks"]),
    }


def answer_node(state: GraphState) -> GraphState:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    context = state["context"]

    user_prompt = build_user_prompt(
        question=question,
        context=context,
        chat_history=chat_history
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer
    }


# =========================================================
# GRAPH
# =========================================================

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("build_context", build_context_node)
    builder.add_node("answer", answer_node)

    builder.set_entry_point("rewrite_query")

    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "build_context")
    builder.add_edge("build_context", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


graph = build_graph()


# =========================================================
# PUBLIC API
# =========================================================

def run_chat_graph(
    question: str,
    chat_history: Optional[List[Dict]] = None,
    return_context: bool = False
) -> Dict:
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
    }

    result = graph.invoke(initial_state)

    output = {
        "question": question,
        "rewritten_query": result.get("rewritten_query"),
        "answer": result.get("answer"),
        "chunk_ids": result.get("chunk_ids", []),
        "sources": result.get("sources", []),
    }

    if return_context:
        output["context"] = result.get("context", "")

    return output