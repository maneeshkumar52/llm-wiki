"""Knowledge graph extraction, storage, and visualization for the local LLM Wiki."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ollama_client import OllamaClient
from utils import ensure_directory, setup_logging, slugify, truncate_text, utc_now_iso

LOGGER = setup_logging()

ENTITY_TYPES = {"person", "team", "process", "tool", "policy", "concept"}
RELATION_TYPES = {
    "owns",
    "manages",
    "uses",
    "requires",
    "part_of",
    "describes",
    "scheduled_at",
    "depends_on",
}

ENTITY_COLORS: dict[str, str] = {
    "person": "#2c7a69",
    "team": "#1a5c4e",
    "process": "#d4a843",
    "tool": "#4a90a4",
    "policy": "#c75b39",
    "concept": "#7b68a8",
}
DEFAULT_NODE_COLOR = "#555555"

EXTRACTION_PROMPT = """Analyze the following text from the document "{source_name}" and extract structured knowledge.

Return a valid JSON object with exactly two keys:
- "entities": an array of objects, each with "name" (string), "type" (one of: person, team, process, tool, policy, concept), and "description" (a one-sentence summary)
- "relationships": an array of objects, each with "source" (entity name), "target" (entity name), "relation" (one of: owns, manages, uses, requires, part_of, describes, scheduled_at, depends_on), and "evidence" (the phrase from the text supporting this relationship)

Rules:
- Only extract entities and relationships clearly stated or strongly implied in the text
- Use the exact entity name as it appears in the text
- Each entity should appear at most once in the entities array
- Return ONLY the JSON object, no other text

Text:
{text}"""

SYNTHESIS_PROMPT = """You are a wiki editor. Write a well-structured wiki article summarizing the following source document "{source_name}".

The article should:
- Start with a one-paragraph overview
- Use clear markdown section headers (##)
- Highlight key facts, policies, people, and processes
- Be written in professional encyclopedic tone
- Include a "Key Topics" section listing the main subjects covered
- Be 200-400 words

Source content:
{content}"""

ENTITY_PAGE_PROMPT = """Write a short wiki article (100-200 words) about "{entity_name}" ({entity_type}) based only on the following evidence from the knowledge base.

The article should summarize what is known about this entity from the available sources.
Mention which documents reference it. Use professional encyclopedic tone.

Evidence:
{evidence}"""


@dataclass
class Entity:
    """A knowledge graph entity extracted from workspace documents."""

    name: str
    entity_type: str
    description: str = ""
    mentions: list[dict[str, str]] = field(default_factory=list)

    def key(self) -> str:
        """Stable deduplication key."""
        return slugify(self.name)


@dataclass
class Relationship:
    """A directed edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    evidence: str = ""
    source_doc: str = ""


@dataclass
class GraphData:
    """Full knowledge graph state for a workspace."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    built_at: str = ""
    source_count: int = 0


class KnowledgeGraphStore:
    """Persist and query the workspace knowledge graph."""

    def __init__(self, graph_dir: Path) -> None:
        self.graph_dir = ensure_directory(graph_dir)
        self.graph_path = self.graph_dir / "graph.json"
        self.entity_pages_dir = ensure_directory(self.graph_dir / "entities")

    def save(self, data: GraphData) -> None:
        """Persist the graph to disk."""
        payload = {
            "built_at": data.built_at,
            "source_count": data.source_count,
            "entities": [asdict(e) for e in data.entities],
            "relationships": [asdict(r) for r in data.relationships],
        }
        self.graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> GraphData:
        """Load a persisted graph from disk."""
        if not self.graph_path.exists():
            return GraphData()
        raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        entities = [Entity(**e) for e in raw.get("entities", [])]
        relationships = [Relationship(**r) for r in raw.get("relationships", [])]
        return GraphData(
            entities=entities,
            relationships=relationships,
            built_at=str(raw.get("built_at", "")),
            source_count=int(raw.get("source_count", 0)),
        )

    def exists(self) -> bool:
        """Return True if a graph has been built."""
        return self.graph_path.exists()

    def write_entity_page(self, entity: Entity, content: str) -> Path:
        """Write a synthesized entity wiki page."""
        target = self.entity_pages_dir / f"{entity.key()}.md"
        target.write_text(content, encoding="utf-8")
        return target

    def list_entity_pages(self) -> list[Path]:
        """Return all generated entity wiki pages."""
        return sorted(self.entity_pages_dir.glob("*.md"))

    def get_stats(self) -> dict[str, Any]:
        """Return graph summary statistics."""
        data = self.load()
        type_counts: dict[str, int] = {}
        for e in data.entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

        cross_doc_entities = 0
        for e in data.entities:
            sources = {m.get("source", "") for m in e.mentions}
            if len(sources) > 1:
                cross_doc_entities += 1

        return {
            "entity_count": len(data.entities),
            "relationship_count": len(data.relationships),
            "entity_type_counts": type_counts,
            "cross_document_entities": cross_doc_entities,
            "entity_page_count": len(self.list_entity_pages()),
            "built_at": data.built_at,
            "source_count": data.source_count,
        }


class KnowledgeGraphBuilder:
    """Extract entities and relationships from document chunks using LLM."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        chat_model: str,
        temperature: float = 0.05,
    ) -> None:
        self.ollama_client = ollama_client
        self.chat_model = chat_model
        self.temperature = temperature

    def extract_from_chunks(
        self,
        chunks: list[str],
        source_name: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from a set of document chunks."""
        combined_text = "\n\n".join(chunks[:6])
        prompt = EXTRACTION_PROMPT.format(
            source_name=source_name,
            text=truncate_text(combined_text, max_chars=3000),
        )
        try:
            raw = self.ollama_client.generate_answer(
                model=self.chat_model,
                question=prompt,
                context_blocks=[],
                system_prompt="You are a knowledge extraction assistant. Return only valid JSON.",
                temperature=self.temperature,
            )
            parsed = self._parse_json(raw)
        except Exception:
            LOGGER.exception("Entity extraction failed for %s", source_name)
            return [], []

        entities: list[Entity] = []
        for entry in parsed.get("entities", []):
            name = str(entry.get("name", "")).strip()
            etype = str(entry.get("type", "concept")).lower()
            if not name:
                continue
            if etype not in ENTITY_TYPES:
                etype = "concept"
            entities.append(
                Entity(
                    name=name,
                    entity_type=etype,
                    description=str(entry.get("description", "")),
                    mentions=[{"source": source_name, "context": truncate_text(combined_text, 200)}],
                )
            )

        relationships: list[Relationship] = []
        entity_names = {e.name.lower() for e in entities}
        for entry in parsed.get("relationships", []):
            src = str(entry.get("source", "")).strip()
            tgt = str(entry.get("target", "")).strip()
            rel = str(entry.get("relation", "")).lower()
            if not src or not tgt:
                continue
            if src.lower() not in entity_names or tgt.lower() not in entity_names:
                continue
            if rel not in RELATION_TYPES:
                rel = "describes"
            relationships.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relation=rel,
                    evidence=str(entry.get("evidence", "")),
                    source_doc=source_name,
                )
            )

        LOGGER.info(
            "Extracted %d entities and %d relationships from %s",
            len(entities),
            len(relationships),
            source_name,
        )
        return entities, relationships

    def synthesize_source_page(
        self,
        source_name: str,
        chunks: list[str],
    ) -> str:
        """Generate a rich wiki-style summary for a source document."""
        combined = "\n\n".join(chunks[:8])
        prompt = SYNTHESIS_PROMPT.format(
            source_name=source_name,
            content=truncate_text(combined, max_chars=3500),
        )
        try:
            result = self.ollama_client.generate_answer(
                model=self.chat_model,
                question=prompt,
                context_blocks=[],
                system_prompt="You are a wiki editor. Write clear, structured markdown.",
                temperature=self.temperature,
            )
            return result.strip()
        except Exception:
            LOGGER.exception("Wiki synthesis failed for %s", source_name)
            return ""

    def synthesize_entity_page(self, entity: Entity) -> str:
        """Generate a wiki page for a single entity."""
        evidence_parts: list[str] = []
        for mention in entity.mentions:
            source = mention.get("source", "unknown")
            context = mention.get("context", "")
            if context:
                evidence_parts.append(f"From {source}: {context}")
        evidence = "\n\n".join(evidence_parts) or "No direct evidence available."

        prompt = ENTITY_PAGE_PROMPT.format(
            entity_name=entity.name,
            entity_type=entity.entity_type,
            evidence=truncate_text(evidence, max_chars=2500),
        )
        try:
            result = self.ollama_client.generate_answer(
                model=self.chat_model,
                question=prompt,
                context_blocks=[],
                system_prompt="You are a wiki editor. Write clear, structured markdown.",
                temperature=self.temperature,
            )
            return result.strip()
        except Exception:
            LOGGER.exception("Entity page synthesis failed for %s", entity.name)
            return f"# {entity.name}\n\nType: {entity.entity_type}\n\n{entity.description}"

    @staticmethod
    def merge_graph_data(
        existing: GraphData,
        new_entities: list[Entity],
        new_relationships: list[Relationship],
    ) -> GraphData:
        """Merge new extraction results into the existing graph, deduplicating entities."""
        entity_map: dict[str, Entity] = {}
        for e in existing.entities:
            entity_map[e.key()] = e
        for e in new_entities:
            key = e.key()
            if key in entity_map:
                existing_entity = entity_map[key]
                seen_sources = {m.get("source") for m in existing_entity.mentions}
                for m in e.mentions:
                    if m.get("source") not in seen_sources:
                        existing_entity.mentions.append(m)
                if e.description and not existing_entity.description:
                    existing_entity.description = e.description
            else:
                entity_map[key] = e

        rel_keys: set[str] = set()
        merged_rels: list[Relationship] = []
        for r in [*existing.relationships, *new_relationships]:
            key = f"{slugify(r.source)}|{r.relation}|{slugify(r.target)}"
            if key not in rel_keys:
                rel_keys.add(key)
                merged_rels.append(r)

        return GraphData(
            entities=list(entity_map.values()),
            relationships=merged_rels,
            built_at=utc_now_iso(),
            source_count=existing.source_count,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Extract the first JSON object from an LLM response."""
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def render_knowledge_graph_html(
    data: GraphData,
    height: str = "520px",
    width: str = "100%",
) -> str:
    """Render the knowledge graph as a standalone HTML string using pyvis."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        bgcolor="#f7f1e8",
        font_color="#1d2b28",
        directed=True,
        select_menu=False,
        filter_menu=False,
    )
    net.barnes_hut(
        gravity=-4000,
        central_gravity=0.35,
        spring_length=120,
        spring_strength=0.02,
        damping=0.15,
    )

    entity_ids: dict[str, int] = {}
    for idx, entity in enumerate(data.entities):
        node_id = idx
        entity_ids[entity.key()] = node_id
        color = ENTITY_COLORS.get(entity.entity_type, DEFAULT_NODE_COLOR)
        label = entity.name
        title = (
            f"<b>{entity.name}</b><br>"
            f"Type: {entity.entity_type}<br>"
            f"Mentions: {len(entity.mentions)} source(s)<br>"
            f"{entity.description}"
        )
        source_count = len({m.get("source", "") for m in entity.mentions})
        size = 18 + (source_count * 6)
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
            font={"size": 13, "color": "#1d2b28", "face": "IBM Plex Sans, sans-serif"},
            borderWidth=2,
            borderWidthSelected=3,
        )

    for rel in data.relationships:
        src_key = slugify(rel.source)
        tgt_key = slugify(rel.target)
        if src_key in entity_ids and tgt_key in entity_ids:
            net.add_edge(
                entity_ids[src_key],
                entity_ids[tgt_key],
                title=f"{rel.relation}: {rel.evidence}" if rel.evidence else rel.relation,
                label=rel.relation,
                color={"color": "#8a8172", "highlight": "#183530"},
                font={"size": 10, "color": "#726754", "align": "middle"},
                arrows="to",
                smooth={"type": "curvedCW", "roundness": 0.15},
            )

    html = net.generate_html()
    return html
