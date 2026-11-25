"""LangGraph RAG workflow: retrieve evidence, then generate a grounded answer."""
from __future__ import annotations
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from cybersage.retrieval import retrieve_relevant_chunks
from cybersage.llm import generate_answer


class ChatState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    retrieved_chunks: list[dict[str, Any]]
    answer: str


def retrieve_node(state: ChatState) -> dict:
    return {"retrieved_chunks": retrieve_relevant_chunks(state["question"])}


def generate_node(state: ChatState) -> dict:
    answer = generate_answer(
        question=state["question"],
        retrieved_chunks=state.get("retrieved_chunks", []),
        history=state.get("history", []),
    )
    return {"answer": answer}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_documents", retrieve_node)
    graph.add_node("generate_grounded_answer", generate_node)
    graph.set_entry_point("retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_grounded_answer")
    graph.add_edge("generate_grounded_answer", END)
    return graph.compile()


CHAT_GRAPH = build_chat_graph()


def answer_question(question: str, history: list[dict[str, str]] | None = None) -> dict:
    if not question.strip():
        raise ValueError("Please enter a question.")
    result = CHAT_GRAPH.invoke(
        {"question": question.strip(), "history": history or []}
    )
    sources = []
    seen = set()
    for item in result.get("retrieved_chunks", []):
        key = (item.get("source"), item.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append(
                {"source": item.get("source"), "page": item.get("page"), "score": item.get("rrf_score")}
            )
    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "retrieval_count": len(result.get("retrieved_chunks", [])),
    }
