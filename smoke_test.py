"""Smoke test for the local LLM Wiki against a real Ollama runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag_pipeline import RAGPipeline
from utils import build_workspace_config, load_config


QUESTION = "What are the internal support hours?"
EXPECTED_CONTEXT_FRAGMENT = "support desk operates Monday through Friday"


def choose_model(pipeline: RAGPipeline) -> str:
    """Pick a preferred local model for smoke testing."""
    available_models = pipeline.list_available_models()
    preferred_order = ["llama3:latest", "mistral:latest", "deepseek-r1:8b"]
    for preferred in preferred_order:
        if preferred in available_models:
            return preferred
    if not available_models:
        raise RuntimeError("No Ollama models are installed locally.")
    return available_models[0]


def main() -> None:
    """Run the end-to-end smoke test."""
    config = load_config()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        config["paths"]["workspace_root"] = str(temp_root / "workspaces")
        config["paths"]["sample_docs_dir"] = "sample_docs"
        config["embeddings"]["provider"] = "ollama"

        workspace_config = build_workspace_config(config, "Smoke Test")
        pipeline = RAGPipeline(workspace_config)
        model_name = choose_model(pipeline)
        workspace_config["ollama"]["chat_model"] = model_name
        workspace_config["embeddings"]["model"] = model_name
        pipeline = RAGPipeline(workspace_config)

        sample_root = Path(workspace_config["paths"]["sample_docs_dir"])
        ingested = []
        for sample_name in ["company_wiki.txt", "faq.md"]:
            sample_path = sample_root / sample_name
            saved_path = pipeline.save_uploaded_file(sample_name, sample_path.read_bytes())
            ingested.append((sample_name, pipeline.ingest_file(saved_path)))
        if not ingested:
            raise RuntimeError("No sample documents were ingested.")

        answer, chunks = pipeline.query(
            QUESTION,
            chat_model=model_name,
            top_k=4,
        )

        if not answer.strip():
            raise RuntimeError("Ollama returned an empty answer.")
        if not chunks:
            raise RuntimeError("Retrieval returned no context chunks.")
        if not any(EXPECTED_CONTEXT_FRAGMENT in chunk.content for chunk in chunks):
            raise RuntimeError("Expected support-hours context was not retrieved.")

        note_path = pipeline.save_answer_to_wiki(
            QUESTION,
            answer,
            chunks,
            author="Smoke Test",
        )
        if not note_path.exists():
            raise RuntimeError("Learned note was not written to the wiki.")

        # Knowledge graph smoke test
        print("\n--- Knowledge Graph ---")
        stats = pipeline.build_knowledge_graph(chat_model=model_name)
        print(f"Entities: {stats['entity_count']}, Relationships: {stats['relationship_count']}")
        if stats["entity_count"] == 0:
            raise RuntimeError("Knowledge graph extracted zero entities.")

        graph_html = pipeline.get_knowledge_graph_html()
        if not graph_html:
            raise RuntimeError("Knowledge graph HTML is empty.")
        print("Graph HTML generated successfully")

        synthesized = pipeline.synthesize_wiki_pages(chat_model=model_name)
        print(f"Synthesized {synthesized} source wiki pages")

        print("\nSmoke test passed")
        print(f"Model: {model_name}")
        print(f"Question: {QUESTION}")
        print(f"Retrieved sources: {', '.join({chunk.source for chunk in chunks})}")
        print("Answer:")
        print(answer)


if __name__ == "__main__":
    main()
