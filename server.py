import os
import random
import json
import hashlib
import threading
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

AZURE_AI_PROJECT_ENDPOINT = os.getenv(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://foundry--endpoint-resource.services.ai.azure.com/api/projects/foundry--endpoint",
)
AZURE_AI_AGENT_NAME = os.getenv("AZURE_AI_AGENT_NAME", "westgate-agent")
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "40"))
MAX_AGENT_TOOL_ROUNDS = int(os.getenv("MAX_AGENT_TOOL_ROUNDS", "20"))
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "4"))
AGENT_RETRY_BASE_SECONDS = float(os.getenv("AGENT_RETRY_BASE_SECONDS", "1.5"))
AGENT_RETRY_MAX_SECONDS = float(os.getenv("AGENT_RETRY_MAX_SECONDS", "12"))
AGENT_VERSION_CHECK_SECONDS = float(os.getenv("AGENT_VERSION_CHECK_SECONDS", "3"))

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CHAT_PAYLOAD_BYTES", str(2 * 1024 * 1024)))

_project_client = None
_openai_client = None
_agent_snapshot = None
_agent_checked_at = 0
_client_lock = threading.RLock()


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def get_project_client():
    global _project_client

    if _project_client is None:
        _project_client = AIProjectClient(
            endpoint=AZURE_AI_PROJECT_ENDPOINT,
            credential=DefaultAzureCredential(),
            allow_preview=True,
        )

    return _project_client


def get_openai_client():
    global _openai_client

    with _client_lock:
        if _openai_client is None:
            _project_client = get_project_client()
            try:
                refresh_agent_snapshot(force=True)
            except Exception:
                app.logger.exception("Azure AI Agent initial version check failed")
            _openai_client = _project_client.get_openai_client(agent_name=AZURE_AI_AGENT_NAME)

    return _openai_client


def reset_openai_client():
    global _openai_client

    with _client_lock:
        if _openai_client is not None:
            try:
                _openai_client.close()
            except Exception:
                pass
        _openai_client = None


def read_model_field(model, *names):
    for name in names:
        if isinstance(model, dict) and name in model:
            return model.get(name)

        value = getattr(model, name, None)
        if value is not None:
            return value

        if "_" in name:
            camel = name.split("_")[0] + "".join(part.title() for part in name.split("_")[1:])
            if isinstance(model, dict) and camel in model:
                return model.get(camel)

            value = getattr(model, camel, None)
            if value is not None:
                return value

    return None


def normalize_snapshot_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): normalize_snapshot_value(item) for key, item in sorted(value.items())}

    if isinstance(value, (list, tuple)):
        return [normalize_snapshot_value(item) for item in value]

    return str(value)


def latest_agent_version_details(project_client):
    versions = project_client.agents.list_versions(
        AZURE_AI_AGENT_NAME,
        order="desc",
        limit=1,
    )

    for version in versions:
        return version

    return None


def build_agent_snapshot(agent, latest_version):
    fingerprint_source = {
        "agent_name": AZURE_AI_AGENT_NAME,
        "agent_id": read_model_field(agent, "id", "agent_id"),
        "agent_version": read_model_field(agent, "version", "current_version", "latest_version"),
        "agent_updated_at": read_model_field(agent, "updated_at", "modified_at"),
        "latest_version_id": read_model_field(latest_version, "id", "version", "agent_version"),
        "latest_version_created_at": read_model_field(latest_version, "created_at"),
        "latest_version_updated_at": read_model_field(latest_version, "updated_at", "modified_at"),
        "latest_version_etag": read_model_field(latest_version, "etag"),
        "latest_version_metadata": read_model_field(latest_version, "metadata"),
    }
    fingerprint_source = normalize_snapshot_value(fingerprint_source)
    encoded = json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    return {
        "agentName": AZURE_AI_AGENT_NAME,
        "fingerprint": fingerprint,
        "version": fingerprint_source.get("latest_version_id")
        or fingerprint_source.get("agent_version")
        or fingerprint[:12],
        "checkedAt": time.time(),
    }


def fetch_agent_snapshot():
    project_client = get_project_client()
    agent = project_client.agents.get(AZURE_AI_AGENT_NAME)

    try:
        latest_version = latest_agent_version_details(project_client)
    except Exception:
        latest_version = None

    return build_agent_snapshot(agent, latest_version)


def refresh_agent_snapshot(force=False):
    global _agent_snapshot, _agent_checked_at

    now = time.time()
    if not force and _agent_snapshot and now - _agent_checked_at < AGENT_VERSION_CHECK_SECONDS:
        return _agent_snapshot

    with _client_lock:
        now = time.time()
        if not force and _agent_snapshot and now - _agent_checked_at < AGENT_VERSION_CHECK_SECONDS:
            return _agent_snapshot

        previous_fingerprint = _agent_snapshot.get("fingerprint") if _agent_snapshot else None
        snapshot = fetch_agent_snapshot()
        _agent_snapshot = snapshot
        _agent_checked_at = now

        if previous_fingerprint and previous_fingerprint != snapshot.get("fingerprint"):
            reset_openai_client()

        return snapshot


def normalize_messages(messages):
    normalized = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        normalized.append({"role": role, "content": content.strip()})

    return normalized[-MAX_AGENT_TURNS:]


def extract_response_text(response):
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    def from_content(content):
        if not isinstance(content, list):
            return ""

        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        return "\n\n".join(parts).strip()

    for item in reversed(getattr(response, "output", []) or []):
        if getattr(item, "type", None) == "message":
            text = from_content(getattr(item, "content", None))
            if text:
                return text

        nested = getattr(item, "output", None)
        if getattr(nested, "type", None) == "message":
            text = from_content(getattr(nested, "content", None))
            if text:
                return text

    return ""


def build_mcp_approvals(response):
    approvals = []

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "mcp_approval_request":
            continue

        approval = {
            "type": "mcp_approval_response",
            "approval_request_id": item.id,
            "approve": True,
        }

        approvals.append(approval)

    return approvals


def is_rate_limit_error(exc):
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status_code == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "rate_limit" in message or "too many requests" in message


def get_retry_delay(exc, attempt):
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        try:
            return min(float(retry_after), AGENT_RETRY_MAX_SECONDS)
        except (TypeError, ValueError):
            pass

    delay = AGENT_RETRY_BASE_SECONDS * (2 ** attempt)
    jitter = random.uniform(0, 0.45)
    return min(delay + jitter, AGENT_RETRY_MAX_SECONDS)


def create_response_with_retries(client, **request):
    last_error = None

    for attempt in range(MAX_AGENT_RETRIES + 1):
        try:
            return client.responses.create(**request)
        except Exception as exc:
            last_error = exc
            if not is_rate_limit_error(exc) or attempt >= MAX_AGENT_RETRIES:
                raise

            time.sleep(get_retry_delay(exc, attempt))

    raise last_error


def run_agent(messages, previous_response_id=None):
    client = get_openai_client()
    request = {"input": messages}
    if previous_response_id:
        request["previous_response_id"] = previous_response_id

    response = create_response_with_retries(client, **request)

    for _ in range(MAX_AGENT_TOOL_ROUNDS):
        reply = extract_response_text(response)
        if reply:
            return reply, response.id

        approvals = build_mcp_approvals(response)
        if not approvals:
            break

        response = create_response_with_retries(
            client,
            previous_response_id=response.id,
            input=approvals,
        )

    raise RuntimeError("Azure AI Agent returned a response without assistant text.")


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    messages = normalize_messages(payload.get("messages", []))
    previous_response_id = payload.get("previousResponseId")

    if not messages or messages[-1]["role"] != "user":
        return jsonify({"error": "A user message is required."}), 400

    messages = [messages[-1]]

    try:
        reply, response_id = run_agent(messages, previous_response_id)
        return jsonify({"reply": reply, "responseId": response_id})
    except Exception as exc:
        app.logger.exception("Azure AI Agent request failed")
        if is_rate_limit_error(exc):
            return jsonify({
                "error": "Azure is temporarily rate limiting this agent deployment. I retried automatically, but the quota window is still busy. Please send the query again in a moment."
            }), 429
        return jsonify({"error": str(exc)}), 502


@app.get("/api/agent-version")
def agent_version():
    try:
        snapshot = refresh_agent_snapshot()
        return jsonify(snapshot)
    except Exception as exc:
        app.logger.exception("Azure AI Agent version check failed")
        return jsonify({"error": str(exc)}), 502


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=True)
