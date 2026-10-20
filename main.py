import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import APIError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from database import seed_db, list_orders, list_refunds
from graph import build_graph
from schemas import CustomerRequest, ReviewRequest


BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env", override=True)

graph_lock = RLock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_db()
    app.state.graph = build_graph(InMemorySaver())
    yield


app = FastAPI(
    title="LangGraph Refund Agent",
    lifespan=lifespan
)


def require_key():
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise HTTPException(
            503,
            "Add GROQ_API_KEY in .env and restart the server."
        )


def run_graph(graph, value, config):
    try:
        return graph.invoke(value, config=config)

    except APIError as error:
        code = getattr(error, "status_code", None)
        body = getattr(error, "body", None)

        info = (
            body.get("error", body)
            if isinstance(body, dict)
            else {}
        )

        provider_message = (
            str(info.get("message", ""))
            if isinstance(info, dict)
            else ""
        )

        key = os.getenv("GROQ_API_KEY", "").strip()

        if key:
            provider_message = provider_message.replace(
                key,
                "[redacted]"
            )

        provider_message = re.sub(
            r"gsk_[A-Za-z0-9_-]+",
            "[redacted]",
            provider_message
        )[:800]

        messages = {
            400: "Groq rejected the request.",
            401: "Invalid Groq API key. Update .env and restart.",
            403: "Your account does not have access to this model.",
            404: "Model not found. Check GROQ_MODEL in .env.",
            429: "Groq rate limit reached. Wait before retrying."
        }

        detail = messages.get(
            code,
            "Could not reach Groq. Check your connection."
        )

        if provider_message:
            detail += " Groq detail: " + provider_message

        print(f"Groq error {code}: {detail}")

        raise HTTPException(502, detail)

    except Exception:
        raise HTTPException(
            500,
            "Agent execution stopped. Check this request status "
            "before retrying; a demo refund may already "
            "have been recorded."
        )


def format_result(graph, thread_id, result=None):
    snapshot = graph.get_state({
        "configurable": {
            "thread_id": thread_id
        }
    })

    values = snapshot.values

    if not values:
        raise HTTPException(
            404,
            "Request not found. Server restarts clear "
            "in-memory requests; CSV orders and refunds "
            "remain saved."
        )

    pending = "human_review" in snapshot.next

    interrupts = [
        item.value
        for item in (result or {}).get("__interrupt__", ())
    ]

    if pending and not interrupts:
        order = values.get("order") or {}

        interrupts = [{
            "order_id": values.get("order_id"),
            "amount": order.get("amount"),
            "message": values.get("message"),
            "question": "Approve this refund?"
        }]

    if pending:
        status = "awaiting_approval"
    elif snapshot.next:
        status = "needs_retry"
    else:
        status = "completed"

    return {
        "thread_id": thread_id,
        "status": status,
        "interrupts": interrupts,
        "intent": values.get("intent"),
        "order_id": values.get("order_id"),
        "order": values.get("order"),
        "decision": values.get("decision"),
        "detail": values.get("detail"),
        "refund_result": values.get("refund_result"),
        "response": (
            values.get("response")
            if status == "completed"
            else None
        ),
        "next_steps": list(snapshot.next)
    }


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.post("/support")
def support(body: CustomerRequest):
    require_key()
    message = body.message.strip()

    if not message:
        raise HTTPException(
            400,
            "Enter a support message."
        )

    with graph_lock:
        graph = app.state.graph

        config = {
            "configurable": {
                "thread_id": body.thread_id
            }
        }

        if graph.get_state(config).values:
            raise HTTPException(
                409,
                "Request already exists. Check its status "
                "or retry the existing request."
            )

        result = run_graph(
            graph,
            {
                "message": message,
                "request_id": str(uuid.uuid4())
            },
            config
        )

        return format_result(
            graph,
            body.thread_id,
            result
        )


@app.post("/review/{thread_id}")
def review(thread_id: str, body: ReviewRequest):
    require_key()

    with graph_lock:
        graph = app.state.graph

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        snapshot = graph.get_state(config)

        if "human_review" not in snapshot.next:
            raise HTTPException(
                409,
                "No pending human review for this request."
            )

        result = run_graph(
            graph,
            Command(resume={
                "approved": body.approved
            }),
            config
        )

        return format_result(
            graph,
            thread_id,
            result
        )


@app.post("/retry/{thread_id}")
def retry(thread_id: str):
    require_key()

    with graph_lock:
        graph = app.state.graph

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        snapshot = graph.get_state(config)

        if not snapshot.values or not snapshot.next:
            raise HTTPException(
                409,
                "This request has no paused execution to retry."
            )

        if "human_review" in snapshot.next:
            raise HTTPException(
                409,
                "Use Approve or Reject for this pending request."
            )

        result = run_graph(
            graph,
            None,
            config
        )

        return format_result(
            graph,
            thread_id,
            result
        )


@app.get("/threads/{thread_id}")
def request_status(thread_id: str):
    with graph_lock:
        return format_result(
            app.state.graph,
            thread_id
        )


@app.get("/orders")
def orders():
    return {
        "orders": list_orders()
    }


@app.get("/refunds")
def refunds():
    return {
        "refunds": list_refunds()
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_ready": bool(
            os.getenv("GROQ_API_KEY", "").strip()
        ),
        "model": (
            os.getenv("GROQ_MODEL", "").strip()
            or "openai/gpt-oss-20b"
        )
    }