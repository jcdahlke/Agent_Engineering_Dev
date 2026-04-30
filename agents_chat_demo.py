"""
OpenAI Agents SDK — Four Conversation-State Strategies
Hosted with Gradio.

Strategies
----------
1. result.to_input_list()   — app holds the full replay list
2. session (SQLite)         — SDK writes to a local SQLite store
3. conversationId           — OpenAI Conversations API manages server-side history
4. previous_response_id     — lightest: just pass the last response ID + new turn
"""
from __future__ import annotations

import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agents import Agent, Runner
from agents.extensions.memory.async_sqlite_session import AsyncSQLiteSession
from agents.memory.openai_conversations_session import OpenAIConversationsSession

# ─── Agent ────────────────────────────────────────────────────────────────────
agent = Agent(
    name="Demo Assistant",
    instructions="You are a helpful assistant. Keep your answers concise and clear.",
    model="gpt-4o-mini",
)

# ─── Strategy metadata ────────────────────────────────────────────────────────
STRATEGY_OPTIONS = [
    ("1 — result.to_input_list()", "to_input_list"),
    ("2 — session (SQLite)", "session"),
    ("3 — conversationId (Conversations API)", "conversation_id"),
    ("4 — previous_response_id (Responses API)", "previous_response_id"),
]
LABELS = [s[0] for s in STRATEGY_OPTIONS]
LABEL_TO_KEY = {s[0]: s[1] for s in STRATEGY_OPTIONS}

DESCRIPTIONS: dict[str, str] = {
    "to_input_list": (
        "### Strategy 1 — `result.to_input_list()`\n"
        "| | |\n|---|---|\n"
        "| **State lives in** | Your application |\n"
        "| **Best for** | Small chat loops and maximum control |\n"
        "| **What you pass next turn** | The replay-ready history |\n\n"
        "Your app owns every message. After each turn `result.to_input_list()` returns "
        "the complete history as a plain list of input items. You append the new user "
        "message and pass the whole list as `input` to the next `Runner.run()` call."
    ),
    "session": (
        "### Strategy 2 — `session`\n"
        "| | |\n|---|---|\n"
        "| **State lives in** | Your storage + the SDK |\n"
        "| **Best for** | Persistent chat state and resumable runs |\n"
        "| **What you pass next turn** | The same session object |\n\n"
        "`AsyncSQLiteSession` stores conversation history in a local SQLite database. "
        "The SDK reads existing items before each turn and writes new items after it. "
        "You only pass `session=` once — no manual list management required."
    ),
    "conversation_id": (
        "### Strategy 3 — `conversationId`\n"
        "| | |\n|---|---|\n"
        "| **State lives in** | OpenAI Conversations API |\n"
        "| **Best for** | Shared server-managed state across workers or services |\n"
        "| **What you pass next turn** | The same conversation ID + only the new turn |\n\n"
        "`OpenAIConversationsSession` creates a conversation in the OpenAI Conversations API. "
        "History lives entirely on OpenAI's servers. Any worker can resume the same "
        "conversation just by referencing the conversation ID."
    ),
    "previous_response_id": (
        "### Strategy 4 — `previous_response_id`\n"
        "| | |\n|---|---|\n"
        "| **State lives in** | OpenAI Responses API |\n"
        "| **Best for** | Lightest server-managed continuation |\n"
        "| **What you pass next turn** | The last response ID + only the new turn |\n\n"
        "After each turn, `result.last_response_id` is stored locally. On the next "
        "call, pass `previous_response_id=` to `Runner.run()` — OpenAI reconstructs "
        "history server-side from that chain of response IDs. No local message list needed."
    ),
}

# ─── Per-session state (plain dict — avoids Gradio 6 dataclass serialisation) ─
def make_state() -> dict:
    return {
        # Per-strategy display histories — Gradio 6 messages format: list of {"role", "content"}
        "display": {k: [] for k in ["to_input_list", "session", "conversation_id", "previous_response_id"]},
        "history_list": [],        # Strategy 1
        "sqlite_session": None,    # Strategy 2
        "conversations_session": None,  # Strategy 3
        "last_response_id": None,  # Strategy 4
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _key(label: str) -> str:
    return LABEL_TO_KEY[label]


def _state_info(state: dict, strategy_key: str) -> str:
    if strategy_key == "to_input_list":
        n = len(state["history_list"])
        return f"*Stored input items in app memory: **{n}***"
    elif strategy_key == "session":
        sess = state["sqlite_session"]
        if sess is None:
            return "*SQLite session: not started yet*"
        return f"*SQLite session ID: `{sess.session_id}`*"
    elif strategy_key == "conversation_id":
        sess = state["conversations_session"]
        if sess is None:
            return "*Conversation: not started yet*"
        try:
            return f"*Conversation ID: `{sess.session_id}`*"
        except ValueError:
            return "*Conversation ID: initializing…*"
    elif strategy_key == "previous_response_id":
        rid = state["last_response_id"] or "none"
        return f"*Last response ID: `{rid}`*"
    return ""


# ─── Event handlers ───────────────────────────────────────────────────────────
async def send(user_message: str, state: dict, strategy_label: str):
    if not user_message.strip():
        key = _key(strategy_label)
        return state["display"][key], state, "", _state_info(state, key)

    key = _key(strategy_label)

    try:
        if key == "to_input_list":
            # Full history replay — app owns the list
            input_data = state["history_list"] + [{"role": "user", "content": user_message}]
            result = await Runner.run(agent, input=input_data)
            state["history_list"] = result.to_input_list()
            reply = result.final_output

        elif key == "session":
            # SDK reads/writes the SQLite session automatically
            if state["sqlite_session"] is None:
                state["sqlite_session"] = AsyncSQLiteSession(session_id=str(uuid.uuid4()))
            result = await Runner.run(agent, input=user_message, session=state["sqlite_session"])
            reply = result.final_output

        elif key == "conversation_id":
            # Server-managed via OpenAI Conversations API
            if state["conversations_session"] is None:
                state["conversations_session"] = OpenAIConversationsSession()
            result = await Runner.run(
                agent, input=user_message, session=state["conversations_session"]
            )
            reply = result.final_output

        elif key == "previous_response_id":
            # Lightest continuation — only the new user turn + the last response ID
            result = await Runner.run(
                agent,
                input=user_message,
                previous_response_id=state["last_response_id"],
            )
            state["last_response_id"] = result.last_response_id
            reply = result.final_output

        else:
            reply = f"Unknown strategy: {key}"

    except Exception as exc:
        reply = f"**Error:** {exc}"

    state["display"][key] = state["display"][key] + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return state["display"][key], state, "", _state_info(state, key)


async def clear_chat(state: dict, strategy_label: str):
    key = _key(strategy_label)

    if key == "to_input_list":
        state["history_list"] = []
    elif key == "session":
        if state["sqlite_session"] is not None:
            await state["sqlite_session"].clear_session()
            state["sqlite_session"] = None
    elif key == "conversation_id":
        if state["conversations_session"] is not None:
            await state["conversations_session"].clear_session()
            state["conversations_session"] = None
    elif key == "previous_response_id":
        state["last_response_id"] = None

    state["display"][key] = []
    return [], state, _state_info(state, key)


def switch_strategy(strategy_label: str, state: dict):
    key = _key(strategy_label)
    return state["display"][key], DESCRIPTIONS[key], _state_info(state, key)


# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="OpenAI Agents SDK — Conversation State Strategies") as demo:
    chat_state = gr.State(make_state)

    gr.Markdown(
        "# OpenAI Agents SDK\n"
        "## Conversation State Strategies\n"
        "Select a strategy, chat with the agent, and observe how each one manages "
        "multi-turn memory differently. Each strategy keeps its own history — "
        "switching tabs picks up where you left off."
    )

    strategy_radio = gr.Radio(
        choices=LABELS,
        value=LABELS[0],
        label="Choose a conversation strategy",
        interactive=True,
    )

    desc_md = gr.Markdown(DESCRIPTIONS["to_input_list"])

    chatbot = gr.Chatbot(label="Conversation", height=440)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type a message and press Enter…",
            label="",
            scale=9,
            container=False,
            autofocus=True,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary", min_width=80)

    clear_btn = gr.Button("Clear this conversation", variant="secondary", size="sm")

    state_info_md = gr.Markdown(_state_info(make_state(), "to_input_list"))

    # ── Event wiring ──────────────────────────────────────────────────────────
    strategy_radio.change(
        switch_strategy,
        inputs=[strategy_radio, chat_state],
        outputs=[chatbot, desc_md, state_info_md],
    )

    send_inputs = [msg_box, chat_state, strategy_radio]
    send_outputs = [chatbot, chat_state, msg_box, state_info_md]

    send_btn.click(send, inputs=send_inputs, outputs=send_outputs)
    msg_box.submit(send, inputs=send_inputs, outputs=send_outputs)

    clear_btn.click(
        clear_chat,
        inputs=[chat_state, strategy_radio],
        outputs=[chatbot, chat_state, state_info_md],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
