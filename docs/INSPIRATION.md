# Inspiration Mapping

This project was reviewed against four external references and updated so the shipped code reflects their strongest ideas rather than only citing them.

## Karpathy gist

Concepts carried into this repo:

- Three-layer thinking: raw sources, compiled wiki artifacts, and app-level rules in configuration.
- A compounding knowledge flow instead of one-off chat over uploaded files.
- Persistent `index.md` and `log.md` per workspace.
- Query answers that can be saved back into the wiki as durable notes.

How it shows up here:

- Raw sources are stored under each workspace in `data/workspaces/<workspace>/raw/sources`.
- Compiled wiki files are written under `data/workspaces/<workspace>/wiki`.
- Learned answers are stored under `wiki/queries/` and re-ingested into the vector store.
- `index.md` and `log.md` are automatically maintained.

## nashsu/llm_wiki

Concepts carried into this repo:

- Workspace-oriented product UX instead of a single throwaway chat.
- Persistent conversations.
- Save-to-wiki workflow.
- Stronger operational controls and visible provenance.

How it shows up here:

- Per-workspace conversation persistence in `.llm-wiki/chats/`.
- Source-aware answer lens and retrieved-context inspection.
- Transcript export.
- Save-last-answer action for compounding knowledge.

## lucasastorian/llmwiki

Concepts carried into this repo:

- Separation between retrieval, wiki maintenance, and user-facing interface.
- Strong emphasis on citations and compiled wiki behavior.
- Production framing around storage and durable state.

How it shows up here:

- `rag_pipeline.py` handles ingestion, retrieval, wiki artifacts, conversations, learning notes, and exports.
- Inline citation prompting is enforced in the Ollama system prompt.
- Workspace storage is organized so different teams or use cases do not collide.

## NicholasSpisak/second-brain

Concepts carried into this repo:

- Obsidian-style second-brain folder mindset.
- Index and log as durable navigational assets.
- LLM-as-librarian flow for evolving knowledge.
- Strong onboarding and sample-question guidance.

How it shows up here:

- The repo now documents workspaces, wiki artifacts, and sample questions explicitly.
- Source summaries are generated into `wiki/sources/`.
- The app can be used as a team handbook assistant or personal second brain.

## What is intentionally not implemented

The referenced projects include heavyweight features such as browser extensions, MCP servers, graph visualization, deep web research, OCR pipelines, and multi-service cloud backends. This repository stays local-first and single-process by design:

- Streamlit UI
- Local Ollama runtime
- Local Chroma persistence
- Local workspace folders

That tradeoff keeps the app easy to run while still implementing the core local LLM wiki pattern credibly.
