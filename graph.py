"""Minimal LangGraph example: conditional routing to different nodes by topic.

LM Studio exposes an OpenAI-compatible API, so we reuse `ChatOpenAI` and just
point it at the local server instead of OpenAI's cloud endpoint.

Flow:
                         +-> code_chat    -+
    START -> classify_topic -+              +-> END
                         +-> general_chat -+

  1. classify_topic: asks the model to label the user's question with a short
     topic (e.g. "python", "cooking", "general"). Stored in state["topic"].
  2. route_by_topic: a plain Python function (no LLM call) that reads
     state["topic"] and decides which node to go to next.
  3. code_chat / general_chat: two differently-specialized answer nodes.
     Only one of them actually runs per request.

This demonstrates branching: instead of every request taking the same path,
the graph picks a path based on state produced by an earlier node.
"""

import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    """Graph state shared across nodes."""

    messages: Annotated[list, add_messages]
    topic: str


def build_llm() -> ChatOpenAI:
    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    # LM Studio does not validate the API key, but the client requires a non-empty string.
    return ChatOpenAI(base_url=base_url, api_key="lm-studio", model=model, temperature=0.3)


llm = build_llm()


def classify_topic_node(state: State) -> State:
    """Node 1: label the latest user message with a one-word topic."""
    last_user_message = state["messages"][-1]
    classification_prompt = [
        SystemMessage(
            content=(
                "Classify the topic of the user's message in exactly one word "
                "(e.g. python, cooking, history, general). Reply with only that word."
            )
        ),
        last_user_message,
    ]
    response = llm.invoke(classification_prompt)
    topic = response.content.strip().lower()
    return {"topic": topic}


CODE_TOPICS = {"python", "javascript", "code", "programming", "java", "sql", "typescript"}


def route_by_topic(state: State) -> str:
    """Router: pick the next node's name based on state, no LLM call needed."""
    topic = state.get("topic", "general")
    return "code_chat" if topic in CODE_TOPICS else "general_chat"


def code_chat_node(state: State) -> State:
    """Answers coding questions with an example-focused system prompt."""
    system_message = SystemMessage(
        content=(
            "You are an expert software engineer. Answer the user's coding "
            "question clearly and include a short code example when useful."
        )
    )
    response = llm.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


def general_chat_node(state: State) -> State:
    """Answers everything else with a friendly, general-purpose system prompt."""
    topic = state.get("topic", "general")
    system_message = SystemMessage(
        content=f"You are a helpful assistant answering a question about {topic}."
    )
    response = llm.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("classify_topic", classify_topic_node)
graph_builder.add_node("code_chat", code_chat_node)
graph_builder.add_node("general_chat", general_chat_node)
graph_builder.add_edge(START, "classify_topic")
graph_builder.add_conditional_edges(
    "classify_topic",
    route_by_topic,
    {"code_chat": "code_chat", "general_chat": "general_chat"},
)
graph_builder.add_edge("code_chat", END)
graph_builder.add_edge("general_chat", END)

graph = graph_builder.compile()
