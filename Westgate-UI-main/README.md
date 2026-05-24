# Westgate Query Bot

A static chatbot UI backed by a small Flask proxy that connects to the Azure AI Foundry Agent named in your environment.

The backend uses the refreshed hosted-agent endpoint through `project.get_openai_client(agent_name=...)`, so the request reaches the agent runtime and its configured tools instead of calling a plain model endpoint.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if your project endpoint or agent name changes.

## Azure Authentication

This app uses `DefaultAzureCredential`, so sign in with one of the supported Azure identity methods before starting the server. For local development, the usual path is:

```bash
az login
```

The signed-in identity must have access to the Azure AI Foundry project and agent.

## Run

```bash
python3 server.py
```

Open `http://127.0.0.1:5000` and chat with the configured Azure Agent.
