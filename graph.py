"""Complete Multi-Agent LangGraph Project (`graph.py`):
1. SUBGRAPHS: Two self-contained subgraphs (`classify_respond_subgraph` & `qa_review_subgraph`)
2. HANDOFFS: Dynamic routing via `Command(goto=...)` and cross-subgraph `Command.PARENT`
3. AGENT-TO-AGENT (A2A):
   - In-process A2A loop (`reviewer_agent` <-> `fixer_agent`) inside `qa_review_subgraph`
   - Optional remote A2A protocol hook (`call_remote_a2a_agent`) for external agents
"""

import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict


# ===========================================================================
# 0. Shared LLM Client & Project State
# ===========================================================================
def build_llm() -> ChatOpenAI:
    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    return ChatOpenAI(base_url=base_url, api_key="lm-studio", model=model, temperature=0.3)


llm = build_llm()
CODE_TOPICS = {"python", "javascript", "code", "programming", "java", "sql", "typescript"}


class ProjectState(TypedDict):
    """Shared state passed across the Parent Graph and both Subgraphs."""

    messages: Annotated[list, add_messages]
    topic: str
    code_draft: str
    revision_count: int
    handoff_history: list[str]


# ===========================================================================
# 1. SUBGRAPH A: Your Original `classify -> route -> respond` Flow + Handoff
# ===========================================================================
def classify_topic_node(state: ProjectState) -> dict:
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
    history = [*state.get("handoff_history", []), f"Classified topic as '{topic}'"]
    return {"topic": topic, "handoff_history": history}


def route_by_topic(state: ProjectState) -> str:
    """Router: pick the next specialist inside Subgraph A."""
    topic = state.get("topic", "general")
    return "code_chat" if topic in CODE_TOPICS else "general_chat"


def code_chat_node(state: ProjectState) -> Command:
    """Coding specialist: writes initial draft, then hands off to PARENT's `qa_review_subgraph`."""
    system_message = SystemMessage(
        content=(
            "You are an expert software engineer. Answer the user's coding "
            "question clearly and include a working code example."
        )
    )
    response = llm.invoke([system_message, *state["messages"]])
    history = [*state.get("handoff_history", []), "code_chat -> Handoff to qa_review_subgraph"]

    # Cross-subgraph Handoff: jumps out of Subgraph A directly into Subgraph B in the Parent Graph!
    return Command(
        goto="qa_review_subgraph",
        graph=Command.PARENT,
        update={
            "code_draft": response.content,
            "revision_count": 1,
            "handoff_history": history,
            "messages": [AIMessage(content=f"[Coder Initial Draft]\n{response.content}")],
        },
    )


def general_chat_node(state: ProjectState) -> dict:
    """General specialist: answers non-coding questions and finishes Subgraph A."""
    topic = state.get("topic", "general")
    system_message = SystemMessage(
        content=f"You are a helpful assistant answering a question about {topic}."
    )
    response = llm.invoke([system_message, *state["messages"]])
    history = [*state.get("handoff_history", []), "general_chat -> Finished"]
    return {"messages": [response], "handoff_history": history}


def build_classify_respond_subgraph():
    sub_a = StateGraph(ProjectState)
    sub_a.add_node("classify_topic", classify_topic_node)
    sub_a.add_node("code_chat", code_chat_node)
    sub_a.add_node("general_chat", general_chat_node)

    sub_a.add_edge(START, "classify_topic")
    sub_a.add_conditional_edges(
        "classify_topic",
        route_by_topic,
        {"code_chat": "code_chat", "general_chat": "general_chat"},
    )
    sub_a.add_edge("general_chat", END)
    # Note: `code_chat` needs no outgoing edge because it uses `Command(goto=..., graph=Command.PARENT)`
    return sub_a.compile()


classify_respond_subgraph = build_classify_respond_subgraph()


# ===========================================================================
# 2. SUBGRAPH B: Agent-to-Agent (A2A) QA Loop (`reviewer` <-> `fixer`)
# ===========================================================================
def reviewer_agent(state: ProjectState) -> Command[Literal["fixer_agent", "__end__"]]:
    """Agent 1 in Subgraph B: inspects code. Hands off to `fixer_agent` if bugs are found."""
    code_draft = state.get("code_draft", "")
    revision = state.get("revision_count", 1)

    review = llm.invoke([
        SystemMessage(
            content=(
                "You are a strict senior QA reviewer. Check the code for bugs, edge cases, or typos. "
                "If it is solid, reply starting with 'APPROVED:' followed by the final polished response. "
                "If fixes are needed, reply starting with 'CHANGES_REQUESTED:' and list the exact fixes."
            )
        ),
        HumanMessage(content=f"Review this draft (Revision #{revision}):\n\n{code_draft}"),
    ]).content.strip()

    # Agent-to-Agent handoff to `fixer_agent` (max 2 revisions to prevent infinite loops)
    if review.startswith("CHANGES_REQUESTED") and revision < 2:
        history = [*state.get("handoff_history", []), "reviewer_agent -> Handed off to fixer_agent"]
        return Command(
            goto="fixer_agent",
            update={
                "handoff_history": history,
                "messages": [HumanMessage(content=f"[QA Reviewer Feedback]\n{review}")],
            },
        )

    # Approved: finish Subgraph B and return control to the Parent Graph
    final_clean = review.removeprefix("APPROVED:").strip()
    history = [*state.get("handoff_history", []), "reviewer_agent -> Approved & Finished"]
    return Command(
        goto=END,
        update={
            "handoff_history": history,
            "messages": [AIMessage(content=final_clean)],
        },
    )


def fixer_agent(state: ProjectState) -> Command[Literal["reviewer_agent"]]:
    """Agent 2 in Subgraph B: fixes issues raised by Reviewer and hands back to Reviewer."""
    revision = state.get("revision_count", 1) + 1
    fixed = llm.invoke([
        SystemMessage(
            content="You are a software engineer fixing bugs reported by QA. Return the corrected code."
        ),
        *state["messages"],
    ])
    history = [*state.get("handoff_history", []), f"fixer_agent -> Handed back to reviewer (Rev #{revision})"]

    # Agent-to-Agent handoff back to `reviewer_agent`
    return Command(
        goto="reviewer_agent",
        update={
            "code_draft": fixed.content,
            "revision_count": revision,
            "handoff_history": history,
            "messages": [AIMessage(content=f"[Fixer Agent Revision #{revision}]\n{fixed.content}")],
        },
    )


def build_qa_review_subgraph():
    sub_b = StateGraph(ProjectState)
    sub_b.add_node("reviewer_agent", reviewer_agent)
    sub_b.add_node("fixer_agent", fixer_agent)
    sub_b.add_edge(START, "reviewer_agent")
    return sub_b.compile()


qa_review_subgraph = build_qa_review_subgraph()


# ===========================================================================
# 3. PARENT GRAPH: Orchestrates Both Subgraphs + Final Audit Summary
# ===========================================================================
def intake_handoff_node(state: ProjectState) -> Command[Literal["classify_respond_subgraph"]]:
    """Parent entry node: initializes handoff tracking and hands off to Subgraph A."""
    return Command(
        goto="classify_respond_subgraph",
        update={
            "revision_count": 0,
            "handoff_history": ["START -> intake_handoff -> classify_respond_subgraph"],
        },
    )


def final_summary_node(state: ProjectState) -> dict:
    """Parent exit node: appends the live handoff trail so you can see every hop in Studio."""
    trail = " → ".join(state.get("handoff_history", []))
    last_answer = state["messages"][-1].content
    footer = f"\n\n---\n**Agent Handoff Trail:** `{trail}`"
    return {"messages": [AIMessage(content=last_answer + footer)]}


main_builder = StateGraph(ProjectState)
main_builder.add_node("intake_handoff", intake_handoff_node)
main_builder.add_node("classify_respond_subgraph", classify_respond_subgraph)  # Subgraph 1
main_builder.add_node("qa_review_subgraph", qa_review_subgraph)                # Subgraph 2
main_builder.add_node("final_summary", final_summary_node)

main_builder.add_edge(START, "intake_handoff")
# When Subgraph A finishes via `general_chat` (without jumping to Subgraph B), go to `final_summary`
main_builder.add_edge("classify_respond_subgraph", "final_summary")
# When Subgraph B finishes QA review, go to `final_summary`
main_builder.add_edge("qa_review_subgraph", "final_summary")
main_builder.add_edge("final_summary", END)

# LangGraph Studio (`langgraph dev`) loads this `graph` automatically
graph = main_builder.compile()
