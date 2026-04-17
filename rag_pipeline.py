"""Document ingestion, retrieval, wiki artifact management, and conversation persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from knowledge_graph import (
    GraphData,
    KnowledgeGraphBuilder,
    KnowledgeGraphStore,
    render_knowledge_graph_html,
)
from ollama_client import OllamaClient
from utils import (
    chunk_text,
    ensure_directory,
    hash_text,
    setup_logging,
    slugify,
    truncate_text,
    utc_now_iso,
)


LOGGER = setup_logging()


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by vector search."""

    content: str
    source: str
    chunk_index: int
    distance: float
    relevance: float
    citation: str


@dataclass(slots=True)
class DocumentRecord:
    """Metadata for an ingested source document."""

    source: str
    file_type: str
    chunk_count: int
    size_bytes: int
    ingested_at: str
    kind: str
    storage_path: str


@dataclass(slots=True)
class ConversationRecord:
    """A persisted chat conversation."""

    conversation_id: str
    title: str
    updated_at: str
    message_count: int


class EmbeddingService:
    """Embedding provider that supports local transformer or Ollama embeddings."""

    def __init__(self, config: dict[str, Any], ollama_client: OllamaClient) -> None:
        self.provider = str(config.get("provider", "sentence_transformers"))
        self.model_name = str(config.get("model", "all-MiniLM-L6-v2"))
        self.ollama_client = ollama_client
        self._model: Any | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        if self.provider == "sentence_transformers":
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers is not installed. Install it before using that embedding provider."
                    ) from exc
                self._model = SentenceTransformer(self.model_name)
            return self._model.encode(texts, normalize_embeddings=True).tolist()

        if self.provider == "ollama":
            return self.ollama_client.embed(self.model_name, texts)

        raise ValueError(
            "Unsupported embedding provider. Use 'sentence_transformers' or 'ollama'."
        )


class RAGPipeline:
    """Workspace-aware local wiki with RAG, learning notes, and chat persistence."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.workspace = config.get("workspace", {})
        self.workspace_name = str(self.workspace.get("name", "General"))
        self.paths = config["paths"]
        self.chunking = config["chunking"]
        self.retrieval = config["retrieval"]
        self.ollama = config["ollama"]
        self.prompts = config["prompts"]

        workspace_root = ensure_directory(
            self.paths.get("workspace_root", "data/workspaces/default")
        )
        self.workspace_root = workspace_root
        self.documents_dir = ensure_directory(self.paths["documents_dir"])
        self.wiki_dir = ensure_directory(self.paths.get("wiki_dir", workspace_root / "wiki"))
        self.source_pages_dir = ensure_directory(self.wiki_dir / "sources")
        self.query_pages_dir = ensure_directory(self.wiki_dir / "queries")
        self.exports_dir = ensure_directory(
            self.paths.get("exports_dir", workspace_root / "exports")
        )
        self.conversations_dir = ensure_directory(
            self.paths.get("conversations_dir", workspace_root / ".llm-wiki" / "chats")
        )
        self.sample_docs_dir = Path(self.paths.get("sample_docs_dir", "sample_docs"))

        manifest_name = self.paths.get("manifest_name", "manifest.json")
        self.manifest_path = self.documents_dir / manifest_name
        self.index_path = self.wiki_dir / "index.md"
        self.log_path = self.wiki_dir / "log.md"
        self._ensure_workspace_files()

        self.ollama_client = OllamaClient(
            base_url=self.ollama["base_url"],
            timeout=int(self.ollama.get("timeout_seconds", 120)),
        )
        self.embedding_service = EmbeddingService(config["embeddings"], self.ollama_client)
        self.chroma_client = chromadb.PersistentClient(
            path=str(ensure_directory(self.paths["vector_db_dir"])),
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl="chroma_telemetry.NoOpTelemetryClient",
                chroma_telemetry_impl="chroma_telemetry.NoOpTelemetryClient",
            ),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.paths.get(
                "collection_name", f"wiki_chunks_{slugify(self.workspace_name)}"
            ),
            metadata={
                "description": f"Local LLM Wiki chunks for {self.workspace_name}"
            },
        )

        self.graph_store = KnowledgeGraphStore(self.wiki_dir / "graph")

    def check_ollama(self) -> bool:
        """Check whether Ollama is reachable."""
        return self.ollama_client.health_check()

    def list_available_models(self) -> list[str]:
        """List models currently available in Ollama."""
        return self.ollama_client.list_models()

    def save_uploaded_file(self, file_name: str, data: bytes) -> Path:
        """Persist uploaded files into the workspace raw source folder."""
        target = self.documents_dir / Path(file_name).name
        target.write_bytes(data)
        return target

    def ingest_file(self, file_path: str | Path, kind: str = "source") -> int:
        """Ingest a supported document into the vector database and wiki index."""
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        text = self._read_document(path)
        chunks = chunk_text(
            text,
            chunk_size=int(self.chunking["chunk_size"]),
            chunk_overlap=int(self.chunking["chunk_overlap"]),
        )
        if not chunks:
            raise ValueError(f"No readable content found in {path.name}")

        self._delete_source_chunks(path.name)
        embeddings = self.embedding_service.embed_texts(chunks)

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            ids.append(hash_text(f"{path.name}:{index}:{chunk}"))
            metadatas.append({"source": path.name, "chunk_index": index, "kind": kind})

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        record = DocumentRecord(
            source=path.name,
            file_type=path.suffix.lower().lstrip("."),
            chunk_count=len(chunks),
            size_bytes=path.stat().st_size,
            ingested_at=utc_now_iso(),
            kind=kind,
            storage_path=str(path),
        )
        self._upsert_manifest_record(record)
        if kind == "source":
            self._write_source_summary(record, chunks)
            self._append_log(
                "ingest", f"Ingested source `{path.name}` into {len(chunks)} chunks."
            )
        else:
            self._append_log("learn", f"Saved learned note `{path.name}` back into the wiki.")
        self._refresh_index()
        LOGGER.info("Ingested %s chunks from %s", len(chunks), path.name)
        return len(chunks)

    def ingest_sample_documents(self) -> list[tuple[str, int]]:
        """Ingest all bundled sample documents into the current workspace."""
        if not self.sample_docs_dir.exists():
            raise FileNotFoundError(
                f"Sample docs directory not found: {self.sample_docs_dir}"
            )

        results: list[tuple[str, int]] = []
        for sample_path in sorted(self.sample_docs_dir.iterdir()):
            if not sample_path.is_file() or sample_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            saved_path = self.save_uploaded_file(sample_path.name, sample_path.read_bytes())
            results.append((sample_path.name, self.ingest_file(saved_path)))
        return results

    def retrieve_chunks(
        self,
        question: str,
        top_k: int | None = None,
        source_filters: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and rerank the most relevant chunks for a question."""
        if self.collection.count() == 0:
            raise ValueError("The knowledge base is empty. Upload and ingest documents first.")

        question_embedding = self.embedding_service.embed_texts([question])[0]
        where_clause = None
        if source_filters:
            where_clause = {"source": {"$in": source_filters}}

        requested_results = int(top_k or self.retrieval["top_k"])
        available_results = self.collection.count()
        if where_clause:
            available_results = len(self.collection.get(where=where_clause).get("ids", []))
        if available_results == 0:
            raise ValueError("No matching context found for the selected document scope.")

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=min(max(requested_results * 2, requested_results), available_results),
            include=["documents", "metadatas", "distances"],
            where=where_clause,
        )

        question_terms = self._tokenize(question)
        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            lexical_overlap = self._lexical_overlap(
                question_terms, self._tokenize(document)
            )
            semantic_relevance = 1.0 / (1.0 + float(distance))
            relevance = (semantic_relevance * 0.7) + (lexical_overlap * 0.3)
            source_name = str(metadata.get("source", "unknown"))
            chunk_index = int(metadata.get("chunk_index", 0))
            chunks.append(
                RetrievedChunk(
                    content=document,
                    source=source_name,
                    chunk_index=chunk_index,
                    distance=float(distance),
                    relevance=relevance,
                    citation=f"{source_name}#chunk-{chunk_index}",
                )
            )

        reranked = sorted(chunks, key=lambda item: item.relevance, reverse=True)
        return reranked[:requested_results]

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a grounded answer from already retrieved chunks."""
        if not retrieved_chunks:
            raise ValueError("No retrieved chunks are available for answer generation.")

        context_blocks = self._build_context_blocks(retrieved_chunks)
        return self.ollama_client.generate_answer(
            model=chat_model or self.ollama["chat_model"],
            question=question,
            context_blocks=context_blocks,
            system_prompt=self.prompts["system_prompt"],
            temperature=float(
                temperature if temperature is not None else self.ollama.get("temperature", 0.1)
            ),
        )

    def stream_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_model: str | None = None,
        temperature: float | None = None,
    ):
        """Stream a grounded answer token-by-token from Ollama."""
        if not retrieved_chunks:
            raise ValueError("No retrieved chunks are available for answer generation.")

        context_blocks = self._build_context_blocks(retrieved_chunks)
        return self.ollama_client.stream_answer(
            model=chat_model or self.ollama["chat_model"],
            question=question,
            context_blocks=context_blocks,
            system_prompt=self.prompts["system_prompt"],
            temperature=float(
                temperature if temperature is not None else self.ollama.get("temperature", 0.1)
            ),
        )

    def query(
        self,
        question: str,
        chat_model: str | None = None,
        top_k: int | None = None,
        source_filters: list[str] | None = None,
        temperature: float | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Retrieve context and generate a grounded answer."""
        retrieved_chunks = self.retrieve_chunks(
            question, top_k=top_k, source_filters=source_filters
        )
        answer = self.generate_answer(
            question,
            retrieved_chunks,
            chat_model=chat_model,
            temperature=temperature,
        )
        self._append_log("query", f"Asked question: {truncate_text(question, 100)}")
        return answer, retrieved_chunks

    def save_answer_to_wiki(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        author: str,
    ) -> Path:
        """Save a useful answer into the wiki and re-ingest it as a learned note."""
        timestamp = utc_now_iso().replace(":", "-")
        file_name = f"query-{timestamp}.md"
        target = self.query_pages_dir / file_name
        sources = []
        for chunk in retrieved_chunks:
            if chunk.source not in sources:
                sources.append(chunk.source)

        content = [
            f"# Learned Answer: {question}",
            "",
            f"- workspace: {self.workspace_name}",
            f"- author: {author}",
            f"- created_at: {utc_now_iso()}",
            f"- sources: {', '.join(sources)}",
            "",
            "## Question",
            question,
            "",
            "## Answer",
            answer,
            "",
            "## Evidence",
        ]
        for chunk in retrieved_chunks:
            content.extend([f"### {chunk.citation}", chunk.content, ""])
        target.write_text("\n".join(content).strip() + "\n", encoding="utf-8")
        self.ingest_file(target, kind="learned_note")
        return target

    def export_chat_transcript(
        self,
        messages: list[dict[str, str]],
        conversation_title: str,
        owner: str,
    ) -> Path:
        """Export the current chat transcript to markdown."""
        timestamp = utc_now_iso().replace(":", "-")
        target = self.exports_dir / f"chat-{slugify(conversation_title)}-{timestamp}.md"
        lines = [
            f"# Transcript: {conversation_title}",
            "",
            f"- workspace: {self.workspace_name}",
            f"- owner: {owner}",
            f"- exported_at: {utc_now_iso()}",
            "",
        ]
        for message in messages:
            role = str(message.get("role", "assistant")).title()
            content = str(message.get("content", ""))
            lines.extend([f"## {role}", content, ""])
        target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        self._append_log("export", f"Exported transcript `{target.name}`.")
        return target

    def save_conversation(
        self,
        conversation_id: str,
        title: str,
        messages: list[dict[str, str]],
    ) -> Path:
        """Persist a conversation to disk."""
        target = self.conversations_dir / f"{conversation_id}.json"
        payload = {
            "conversation_id": conversation_id,
            "title": title,
            "updated_at": utc_now_iso(),
            "messages": messages,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def load_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Load one conversation from disk."""
        target = self.conversations_dir / f"{conversation_id}.json"
        if not target.exists():
            return {
                "conversation_id": conversation_id,
                "title": "New Conversation",
                "updated_at": utc_now_iso(),
                "messages": [],
            }
        return json.loads(target.read_text(encoding="utf-8"))

    def list_conversations(self) -> list[ConversationRecord]:
        """Return all persisted conversations for the workspace."""
        conversations: list[ConversationRecord] = []
        for file_path in self.conversations_dir.glob("*.json"):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            conversations.append(
                ConversationRecord(
                    conversation_id=str(payload.get("conversation_id", file_path.stem)),
                    title=str(payload.get("title", "Untitled Conversation")),
                    updated_at=str(payload.get("updated_at", utc_now_iso())),
                    message_count=len(payload.get("messages", [])),
                )
            )
        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a persisted conversation."""
        target = self.conversations_dir / f"{conversation_id}.json"
        if target.exists():
            target.unlink()

    def list_ingested_sources(self) -> list[str]:
        """Return distinct source file names present in the vector store."""
        return [record.source for record in self.get_library()]

    def get_library(self) -> list[DocumentRecord]:
        """Return the current document library from the persisted manifest."""
        manifest = self._load_manifest()
        records = [DocumentRecord(**payload) for payload in manifest.values()]
        return sorted(records, key=lambda record: record.ingested_at, reverse=True)

    def get_stats(self) -> dict[str, int]:
        """Return collection and storage statistics for the UI."""
        library = self.get_library()
        return {
            "documents": len([record for record in library if record.kind == "source"]),
            "learned_notes": len(
                [record for record in library if record.kind == "learned_note"]
            ),
            "chunks": self.collection.count(),
            "storage_bytes": sum(record.size_bytes for record in library),
        }

    def get_wiki_index(self) -> str:
        """Return the current wiki index markdown."""
        return self.index_path.read_text(encoding="utf-8")

    def get_wiki_log(self) -> str:
        """Return the current wiki log markdown."""
        return self.log_path.read_text(encoding="utf-8")

    def list_query_pages(self) -> list[Path]:
        """Return saved learned notes from the wiki query folder."""
        return sorted(self.query_pages_dir.glob("*.md"), reverse=True)

    # ------------------------------------------------------------------
    # Knowledge graph and wiki synthesis
    # ------------------------------------------------------------------

    def _get_graph_builder(self, chat_model: str | None = None) -> KnowledgeGraphBuilder:
        """Create a graph builder using the active Ollama client."""
        return KnowledgeGraphBuilder(
            ollama_client=self.ollama_client,
            chat_model=chat_model or self.ollama["chat_model"],
            temperature=float(self.ollama.get("temperature", 0.05)),
        )

    def build_knowledge_graph(self, chat_model: str | None = None) -> dict[str, Any]:
        """Extract entities and relationships from all indexed sources."""
        builder = self._get_graph_builder(chat_model)
        manifest = self._load_manifest()
        source_records = [
            (name, rec) for name, rec in manifest.items() if rec.get("kind") == "source"
        ]
        if not source_records:
            raise ValueError("No source documents indexed. Upload documents first.")

        graph_data = self.graph_store.load() if self.graph_store.exists() else GraphData()
        for source_name, _ in source_records:
            chunks = self._get_chunks_for_source(source_name)
            if not chunks:
                continue
            entities, relationships = builder.extract_from_chunks(chunks, source_name)
            graph_data = KnowledgeGraphBuilder.merge_graph_data(
                graph_data, entities, relationships
            )

        graph_data.source_count = len(source_records)
        graph_data.built_at = utc_now_iso()
        self.graph_store.save(graph_data)

        for entity in graph_data.entities:
            content = builder.synthesize_entity_page(entity)
            self.graph_store.write_entity_page(entity, content)

        self._append_log(
            "graph",
            f"Built knowledge graph: {len(graph_data.entities)} entities, "
            f"{len(graph_data.relationships)} relationships from {len(source_records)} sources.",
        )
        self._refresh_index()
        return self.graph_store.get_stats()

    def synthesize_wiki_pages(self, chat_model: str | None = None) -> int:
        """Run the LLM synthesis pass over all source documents."""
        builder = self._get_graph_builder(chat_model)
        manifest = self._load_manifest()
        source_records = [
            (name, rec) for name, rec in manifest.items() if rec.get("kind") == "source"
        ]
        synthesized = 0
        for source_name, record in source_records:
            chunks = self._get_chunks_for_source(source_name)
            if not chunks:
                continue
            content = builder.synthesize_source_page(source_name, chunks)
            if not content:
                continue
            target = self.source_pages_dir / f"{Path(source_name).stem}.md"
            header = (
                f"# {Path(source_name).stem}\n\n"
                f"- source_file: {source_name}\n"
                f"- kind: {record.get('kind', 'source')}\n"
                f"- synthesized_at: {utc_now_iso()}\n"
                f"- chunk_count: {record.get('chunk_count', 0)}\n\n"
            )
            target.write_text(header + content + "\n", encoding="utf-8")
            synthesized += 1

        self._append_log("synthesis", f"Synthesized {synthesized} wiki source pages via LLM.")
        self._refresh_index()
        return synthesized

    def get_knowledge_graph_data(self) -> GraphData:
        """Return the current workspace knowledge graph data."""
        return self.graph_store.load()

    def get_knowledge_graph_html(self) -> str:
        """Render the knowledge graph as interactive HTML."""
        data = self.graph_store.load()
        if not data.entities:
            return ""
        return render_knowledge_graph_html(data)

    def get_knowledge_graph_stats(self) -> dict[str, Any]:
        """Return graph summary statistics."""
        return self.graph_store.get_stats()

    def list_entity_pages(self) -> list[Path]:
        """Return all auto-generated entity wiki pages."""
        return self.graph_store.list_entity_pages()

    def _get_chunks_for_source(self, source_name: str) -> list[str]:
        """Retrieve raw chunk texts for a source from the vector store."""
        results = self.collection.get(
            where={"source": source_name},
            include=["documents"],
        )
        return list(results.get("documents", []))

    def delete_source(self, source_name: str) -> None:
        """Delete an indexed item from storage, vector DB, and wiki records."""
        existing = self.collection.get(where={"source": source_name})
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

        manifest = self._load_manifest()
        payload = manifest.get(source_name, {})
        storage_path = Path(str(payload.get("storage_path", self.documents_dir / source_name)))
        if storage_path.exists():
            storage_path.unlink()

        if source_name in manifest:
            del manifest[source_name]
            self._save_manifest(manifest)

        summary_page = self.source_pages_dir / f"{Path(source_name).stem}.md"
        if summary_page.exists():
            summary_page.unlink()

        self._append_log("delete", f"Deleted indexed item `{source_name}`.")
        self._refresh_index()

    def reset(self) -> None:
        """Clear all vectors, workspace docs, wiki pages, and conversations."""
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.paths.get(
                "collection_name", f"wiki_chunks_{slugify(self.workspace_name)}"
            ),
            metadata={
                "description": f"Local LLM Wiki chunks for {self.workspace_name}"
            },
        )
        for folder in [
            self.documents_dir,
            self.source_pages_dir,
            self.query_pages_dir,
            self.exports_dir,
            self.conversations_dir,
        ]:
            for file_path in folder.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        self._save_manifest({})
        self._write_default_index()
        self._write_default_log()
        LOGGER.info("Knowledge base reset complete")

    def _read_document(self, path: Path) -> str:
        """Read a document based on its file extension."""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._read_pdf(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def _delete_source_chunks(self, source_name: str) -> None:
        """Remove old chunks for a source before re-ingesting it."""
        existing = self.collection.get(where={"source": source_name})
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def _build_context_blocks(self, retrieved_chunks: list[RetrievedChunk]) -> list[str]:
        """Build LLM context blocks with stable citation labels."""
        return [
            f"Citation: [{chunk.citation}]\nSource: {chunk.source}\nContent:\n{chunk.content}"
            for chunk in retrieved_chunks
        ]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into normalized terms for light reranking."""
        normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
        return {token for token in normalized.split() if len(token) > 2}

    @staticmethod
    def _lexical_overlap(question_terms: set[str], document_terms: set[str]) -> float:
        """Return normalized term overlap for reranking."""
        if not question_terms:
            return 0.0
        return len(question_terms & document_terms) / len(question_terms)

    def _ensure_workspace_files(self) -> None:
        """Create the workspace manifest, index, and log files if missing."""
        if not self.manifest_path.exists():
            self._save_manifest({})
        if not self.index_path.exists():
            self._write_default_index()
        if not self.log_path.exists():
            self._write_default_log()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load the document manifest from disk."""
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            return {}
        return payload

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        """Persist the document manifest to disk."""
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)

    def _upsert_manifest_record(self, record: DocumentRecord) -> None:
        """Insert or update one document manifest entry."""
        manifest = self._load_manifest()
        manifest[record.source] = {
            "source": record.source,
            "file_type": record.file_type,
            "chunk_count": record.chunk_count,
            "size_bytes": record.size_bytes,
            "ingested_at": record.ingested_at,
            "kind": record.kind,
            "storage_path": record.storage_path,
        }
        self._save_manifest(manifest)

    def _write_source_summary(self, record: DocumentRecord, chunks: list[str]) -> None:
        """Create or update a wiki source summary page."""
        target = self.source_pages_dir / f"{Path(record.source).stem}.md"
        preview = "\n\n".join(f"- {truncate_text(chunk, 220)}" for chunk in chunks[:3])
        lines = [
            f"# {Path(record.source).stem}",
            "",
            f"- source_file: {record.source}",
            f"- kind: {record.kind}",
            f"- ingested_at: {record.ingested_at}",
            f"- chunk_count: {record.chunk_count}",
            f"- storage_path: {record.storage_path}",
            "",
            "## Summary",
            "This page was generated automatically during document ingest. It gives the team a stable wiki page to browse even before deeper synthesis is added.",
            "",
            "## Highlights",
            preview or "- No preview available.",
            "",
        ]
        target.write_text("\n".join(lines), encoding="utf-8")

    def _write_default_index(self) -> None:
        """Write the initial wiki index file."""
        self.index_path.write_text(
            "\n".join(
                [
                    f"# Index - {self.workspace_name}",
                    "",
                    "This file catalogs generated wiki pages and learned notes.",
                    "",
                    "## Source Pages",
                    "",
                    "## Learned Notes",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_default_log(self) -> None:
        """Write the initial wiki log file."""
        self.log_path.write_text(
            "\n".join(
                [
                    f"# Log - {self.workspace_name}",
                    "",
                    "Chronological record of ingests, queries, exports, and learned notes.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _refresh_index(self) -> None:
        """Regenerate the wiki index from current source, learned, and entity pages."""
        source_items = []
        for page in sorted(self.source_pages_dir.glob("*.md")):
            source_items.append(f"- [{page.stem}](sources/{page.name})")

        learned_items = []
        for page in sorted(self.query_pages_dir.glob("*.md"), reverse=True):
            learned_items.append(f"- [{page.stem}](queries/{page.name})")

        entity_items = []
        for page in sorted(self.graph_store.entity_pages_dir.glob("*.md")):
            entity_items.append(f"- [{page.stem}](graph/entities/{page.name})")

        lines = [
            f"# Index - {self.workspace_name}",
            "",
            "This file catalogs generated wiki pages, learned notes, and entity pages.",
            "",
            "## Source Pages",
            *(source_items or ["- None yet"]),
            "",
            "## Learned Notes",
            *(learned_items or ["- None yet"]),
            "",
            "## Entity Pages",
            *(entity_items or ["- None yet"]),
            "",
        ]
        self.index_path.write_text("\n".join(lines), encoding="utf-8")

    def _append_log(self, action: str, message: str) -> None:
        """Append an operation entry to the workspace log."""
        entry = f"## [{utc_now_iso()}] {action}\n{message}\n\n"
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(entry)
