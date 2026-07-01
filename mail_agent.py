import os
import json
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from imap_tools import MailBox, AND, MailMessage

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = os.getenv("IMAP_FOLDER", "INBOX")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")


def require_env() -> None:
    missing = [
        key for key, value in {
            "IMAP_HOST": IMAP_HOST,
            "IMAP_USER": IMAP_USER,
            "IMAP_PASSWORD": IMAP_PASSWORD,
        }.items() if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def connect() -> MailBox:
    mb = MailBox(IMAP_HOST)  # type: ignore[arg-type]
    mb.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)  # type: ignore[arg-type]
    return mb


def _mail_to_dict(mail: MailMessage) -> dict:
    return {
        "uid": str(mail.uid),
        "date": mail.date.astimezone().strftime("%Y-%m-%d %H:%M"),
        "subject": mail.subject or "(no subject)",
        "sender": mail.from_ or "(unknown sender)",
    }


@tool
def list_unread_emails() -> str:
    """Return JSON array of unread emails with uid, date, subject, sender."""
    try:
        with connect() as mb:
            unread = list(
                mb.fetch(
                    criteria=AND(seen=False),
                    headers_only=True,
                    mark_seen=False,
                )
            )

        if not unread:
            return json.dumps({"ok": True, "emails": [], "message": "No unread messages."})

        return json.dumps(
            {"ok": True, "emails": [_mail_to_dict(m) for m in unread]},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Failed to list unread emails: {e}"})


@tool
def summarize_email(uid: str) -> str:
    """Summarize one email by IMAP UID. Returns JSON with summary text."""
    try:
        with connect() as mb:
            mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)

        if not mail:
            return json.dumps({"ok": False, "error": f"Email UID {uid} not found."})

        body = (mail.text or "").strip()
        if not body and mail.html:
            body = "HTML-only email content present."

        prompt = (
            "Summarize this email in 3-5 bullet points.\n"
            "Include intent, required action, and urgency.\n\n"
            f"Subject: {mail.subject}\n"
            f"From: {mail.from_}\n"
            f"Date: {mail.date}\n\n"
            f"Body:\n{body[:6000]}"
        )

        summary = raw_llm.invoke(prompt).content
        return json.dumps({"ok": True, "uid": uid, "summary": summary}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Failed to summarize UID {uid}: {e}"})


def llm_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def route_after_llm(state: ChatState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


if __name__ == "__main__":
    require_env()

    llm = init_chat_model(CHAT_MODEL, model_provider="ollama").bind_tools(
        [list_unread_emails, summarize_email]
    )
    raw_llm = init_chat_model(CHAT_MODEL, model_provider="ollama")

    tool_node = ToolNode([list_unread_emails, summarize_email])

    builder = StateGraph(ChatState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "end": END})
    builder.add_edge("tools", "llm")

    graph = builder.compile()

    print('Type an instruction or "quit".\n')
    state: ChatState = {"messages": []}

    while True:
        user_message = input("> ").strip()
        if user_message.lower() == "quit":
            break
        if not user_message:
            continue

        state["messages"].append({"role": "user", "content": user_message})
        state = graph.invoke(state)
        print(state["messages"][-1].content, "\n")