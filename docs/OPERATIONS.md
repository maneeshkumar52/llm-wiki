# Operations Guide

## Workspace layout

Each workspace is isolated under:

```text
data/workspaces/<workspace-slug>/
├── raw/sources/           # Uploaded source files
├── chroma/                # Vector store persistence
├── wiki/
│   ├── index.md           # Catalog of source pages and learned notes
│   ├── log.md             # Chronological operation log
│   ├── sources/           # Auto-generated source summary pages
│   └── queries/           # Saved learned answers
├── exports/               # Transcript exports
└── .llm-wiki/chats/       # Persistent conversations
```

## Automated learning

The application now has a basic compounding-learning loop:

- Upload a source document.
- The document is embedded and indexed.
- A source summary page is created under `wiki/sources/`.
- The workspace `index.md` and `log.md` are updated.
- When a useful answer appears in chat, save it back into the wiki.
- That saved answer is re-ingested as a learned note, which makes future retrieval stronger.

## Conversations

Conversations are stored per workspace as JSON files. This gives the app:

- multi-conversation switching
- persistence across app restarts
- exportable transcripts
- a clearer audit trail for team use cases

## Local authentication

Authentication is optional and configured in `config.yaml`.

- `security.enabled: false` keeps local development friction low.
- Set `security.enabled: true` to require sign-in.
- Configure users with `username`, `display_name`, `role`, `password_sha256`, and `workspaces`.

Generate a password hash with:

```bash
python -c "import hashlib; print(hashlib.sha256(b'your-password').hexdigest())"
```

## Verification commands

Automated tests:

```bash
pytest
```

Real local smoke test:

```bash
python smoke_test.py
```

Run the application:

```bash
streamlit run app.py
```

## Production notes

- Keep `data/workspaces/` on persistent storage.
- Do not commit populated workspace data to source control.
- Replace the example auth hash before enabling login in shared environments.
- If you expose the app beyond localhost, place it behind a reverse proxy and network controls.
- For larger corpora, tune `chunk_size` and `top_k` before increasing model size.
