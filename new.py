import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from imap_tools import AND, MailBox

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = os.getenv("IMAP_FOLDER", "INBOX")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def connect():
    mb = MailBox(IMAP_HOST)
    mb.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)
    return mb


@tool
def list_unread_emails() -> str:
    """List unread emails."""
    try:
        with connect() as mb:
            mails = list(
                mb.fetch(
                    AND(seen=False),
                    headers_only=True,
                    mark_seen=False,
                )
            )

        result = []
        for m in mails:
            result.append(
                {
                    "uid": str(m.uid),
                    "from": m.from_,
                    "subject": m.subject,
                    "date": str(m.date),
                }
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        return str(e)


raw_llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
)


@tool
def summarize_email(uid: str) -> str:
    """Summarize an email using its UID."""
    try:
        with connect() as mb:
            mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)

        if mail is None:
            return "Email not found."

        body = mail.text or ""

        prompt = f"""
Summarize this email.

Subject: {mail.subject}
From: {mail.from_}

Body:
{body[:5000]}
"""

        return raw_llm.invoke(prompt).content

    except Exception as e:
        return str(e)


llm = raw_llm.bind_tools(
    [
        list_unread_emails,
        summarize_email,
    ]
)

tool_node = ToolNode(
    [
        list_unread_emails,
        summarize_email,
    ]
)


def chatbot(state: ChatState):
    return {
        "messages": [
            llm.invoke(state["messages"])
        ]
    }


def route(state: ChatState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    route,
    {
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("tools", "chatbot")

graph = builder.compile()


if __name__ == "__main__":
    print("Email Agent")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("> ")

        if q.lower() == "quit":
            break

        state = {
            "messages": [
                HumanMessage(content=q)
            ]
        }

        result = graph.invoke(state)

        print()
        print(result["messages"][-1].content)
        print()
