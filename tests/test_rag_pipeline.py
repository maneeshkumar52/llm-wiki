"""Tests for the retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph import (
    Entity,
    GraphData,
    KnowledgeGraphBuilder,
    KnowledgeGraphStore,
    Relationship,
)
from rag_pipeline import RAGPipeline
from utils import build_workspace_config


def _build_test_config(tmp_path: Path) -> dict:
    base = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "chat_model": "fake-chat-model",
            "temperature": 0.1,
            "timeout_seconds": 10,
        },
        "embeddings": {
            "provider": "ollama",
            "model": "fake-embed-model",
        },
        "chunking": {
            "chunk_size": 120,
            "chunk_overlap": 20,
        },
        "retrieval": {
            "top_k": 3,
        },
        "paths": {
            "workspace_root": str(tmp_path / "workspaces"),
            "sample_docs_dir": str(tmp_path / "samples"),
        },
        "prompts": {
            "system_prompt": "Be precise and cite context.",
        },
    }
    return build_workspace_config(base, "Test Space")


def _fake_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        vectors.append(
            [
                1.0 if "support" in lowered else 0.0,
                1.0 if "release" in lowered else 0.0,
                1.0 if "metrics" in lowered else 0.0,
            ]
        )
    return vectors


def test_ingest_query_save_learning_and_delete_source(tmp_path: Path) -> None:
    """The pipeline should ingest, retrieve, save learned notes, and delete items consistently."""
    config = _build_test_config(tmp_path)
    pipeline = RAGPipeline(config)
    pipeline.embedding_service.embed_texts = _fake_embed  # type: ignore[method-assign]
    pipeline.ollama_client.generate_answer = lambda **_: "Support is Monday to Friday."  # type: ignore[method-assign]

    document_path = Path(config["paths"]["documents_dir"]) / "company.txt"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(
        "The internal support desk operates Monday through Friday from 9:00 AM to 6:00 PM.",
        encoding="utf-8",
    )

    chunk_count = pipeline.ingest_file(document_path)
    assert chunk_count >= 1
    assert pipeline.get_stats()["documents"] == 1
    assert (Path(config["paths"]["wiki_dir"]) / "index.md").exists()

    answer, chunks = pipeline.query("When does the support desk operate?", top_k=2)
    assert answer == "Support is Monday to Friday."
    assert chunks
    assert chunks[0].source == "company.txt"
    assert chunks[0].citation.startswith("company.txt#chunk-")

    note_path = pipeline.save_answer_to_wiki(
        "When does the support desk operate?",
        answer,
        chunks,
        author="Tester",
    )
    assert note_path.exists()
    assert pipeline.get_stats()["learned_notes"] == 1
    assert "Learned Notes" in pipeline.get_wiki_index()

    transcript_path = pipeline.export_chat_transcript(
        [
            {"role": "user", "content": "When does support run?"},
            {"role": "assistant", "content": answer},
        ],
        "Support Ops",
        "Tester",
    )
    assert transcript_path.exists()

    pipeline.delete_source("company.txt")
    assert pipeline.get_stats()["documents"] == 0
    assert note_path.exists()


# ---- Knowledge Graph Tests ----


def test_graph_store_save_load(tmp_path: Path) -> None:
    """KnowledgeGraphStore should persist and reload graph data."""
    store = KnowledgeGraphStore(tmp_path / "graph")
    gd = GraphData(
        entities=[Entity(name="Acme", entity_type="organization", description="A company")],
        relationships=[
            Relationship(source="Acme", target="Bob", relation="employs", evidence="Bob works at Acme", source_doc="doc.txt")
        ],
    )
    store.save(gd)
    loaded = store.load()
    assert loaded is not None
    assert len(loaded.entities) == 1
    assert loaded.entities[0].name == "Acme"
    assert len(loaded.relationships) == 1
    assert loaded.relationships[0].relation == "employs"


def test_graph_store_write_entity_page(tmp_path: Path) -> None:
    """KnowledgeGraphStore should write entity markdown pages."""
    store = KnowledgeGraphStore(tmp_path / "graph")
    entity = Entity(name="Acme Corp", entity_type="organization", description="A company")
    store.write_entity_page(entity, "# Acme Corp\n\nA company that does things.")
    pages = store.list_entity_pages()
    assert len(pages) == 1
    assert "acme-corp" in pages[0].stem
    assert "# Acme Corp" in pages[0].read_text(encoding="utf-8")


def test_graph_merge() -> None:
    """KnowledgeGraphBuilder.merge_graph_data should combine two graphs."""
    e1 = Entity(name="Alice", entity_type="person", description="Engineer")
    e2 = Entity(name="Bob", entity_type="person", description="Manager")
    e3 = Entity(name="Alice", entity_type="person", description="Senior Engineer", mentions=[{"source": "doc2"}])
    r1 = Relationship(source="Alice", target="Bob", relation="reports_to", evidence="org chart", source_doc="doc1")
    r2 = Relationship(source="Alice", target="Bob", relation="collaborates_with", evidence="project notes", source_doc="doc2")

    existing = GraphData(entities=[e1, e2], relationships=[r1])
    merged = KnowledgeGraphBuilder.merge_graph_data(existing, [e3], [r2])
    names = [e.name for e in merged.entities]
    assert "Alice" in names
    assert "Bob" in names
    assert len(merged.relationships) == 2


def test_graph_stats(tmp_path: Path) -> None:
    """KnowledgeGraphStore.get_stats should reflect stored data."""
    store = KnowledgeGraphStore(tmp_path / "graph")

    # Empty state
    stats = store.get_stats()
    assert stats["entity_count"] == 0
    assert stats["relationship_count"] == 0

    # With data
    gd = GraphData(
        entities=[
            Entity(name="X", entity_type="tool", description="A tool", mentions=[{"source": "a.txt"}, {"source": "b.txt"}]),
            Entity(name="Y", entity_type="process", description="A process"),
        ],
        relationships=[
            Relationship(source="X", target="Y", relation="supports", evidence="docs", source_doc="a.txt"),
        ],
    )
    store.save(gd)
    stats = store.get_stats()
    assert stats["entity_count"] == 2
    assert stats["relationship_count"] == 1
    assert stats["cross_document_entities"] == 1
    assert stats["entity_type_counts"]["tool"] == 1
    assert stats["entity_type_counts"]["process"] == 1


def test_pipeline_knowledge_graph_integration(tmp_path: Path) -> None:
    """Pipeline knowledge graph methods should work end-to-end with mocked LLM."""
    config = _build_test_config(tmp_path)
    pipeline = RAGPipeline(config)
    pipeline.embedding_service.embed_texts = _fake_embed  # type: ignore[method-assign]

    # Prepare a document
    doc_path = Path(config["paths"]["documents_dir"]) / "team.txt"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "Alice is the tech lead. Bob is a senior engineer. Alice mentors Bob on architecture.",
        encoding="utf-8",
    )
    pipeline.ingest_file(doc_path)

    # Mock LLM to return structured JSON for extraction
    extraction_json = json.dumps({
        "entities": [
            {"name": "Alice", "type": "person", "description": "Tech lead"},
            {"name": "Bob", "type": "person", "description": "Senior engineer"},
        ],
        "relationships": [
            {"source": "Alice", "target": "Bob", "relation": "mentors", "evidence": "Alice mentors Bob on architecture"},
        ],
    })
    pipeline.ollama_client.generate_answer = lambda **_: extraction_json  # type: ignore[method-assign]

    stats = pipeline.build_knowledge_graph(chat_model="fake-chat-model")
    assert stats["entity_count"] >= 2
    assert stats["relationship_count"] >= 1

    # Graph HTML should be available
    html = pipeline.get_knowledge_graph_html()
    assert html is not None
    assert "Alice" in html

    # Graph data should be retrievable
    gd = pipeline.get_knowledge_graph_data()
    assert len(gd.entities) >= 2
