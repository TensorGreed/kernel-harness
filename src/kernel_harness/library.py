"""Cross-hackathon knowledge library.

Persists what the loop learns so it accumulates across hackathons (the user's
explicit goal). Two layers, per the design:

* **Raw files** — one JSON per entry under ``<root>/<kind>/``. Human-readable,
  editable, version-controllable. Always present.
* **Retrieval index** — semantic-ish lookup feeding ``retrieve_knowledge``.
  Uses **ChromaDB** when importable; otherwise falls back to a dependency-free
  **lexical** scorer over the raw entries. The fallback means a fresh machine (no
  embedding model downloaded yet) still works on day one.

Four entry kinds: ``technique``, ``failed_approach``, ``winning_kernel``,
``hardware_note``. ``persist_entries`` ingests the ``LibraryEntries`` emitted by
the ``update_library`` subagent; ``candidates_for`` returns the readable strings
the ``library_retrieval`` subagent reasons over. IDs are content hashes, so
re-persisting the same lesson is idempotent. See CONTEXT.md → "Knowledge Library".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .subagents import LibraryEntries

_KINDS = ("technique", "failed_approach", "winning_kernel", "hardware_note")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class LibraryEntry:
    kind: str
    title: str
    text: str                       # searchable + display string
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        h = hashlib.sha1(f"{self.kind}|{self.title}|{self.text}".encode()).hexdigest()
        return h[:16]


# --------------------------------------------------------------------------- #
# Retrieval index backends
# --------------------------------------------------------------------------- #
class RetrievalIndex(Protocol):
    def add(self, id: str, text: str) -> None: ...
    def query(self, text: str, k: int) -> list[str]: ...  # returns ids, best first


class LexicalIndex:
    """Dependency-free TF/IDF-ish ranker. Good enough for a small library."""

    def __init__(self) -> None:
        self._docs: dict[str, set[str]] = {}

    def add(self, id: str, text: str) -> None:
        self._docs[id] = set(_TOKEN_RE.findall(text.lower()))

    def query(self, text: str, k: int) -> list[str]:
        q = set(_TOKEN_RE.findall(text.lower()))
        if not q or not self._docs:
            return []
        n = len(self._docs)
        df = {t: sum(t in d for d in self._docs.values()) for t in q}
        scored = []
        for doc_id, tokens in self._docs.items():
            score = sum(
                math.log(1 + n / df[t]) for t in q if t in tokens and df[t]
            )
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored[:k]]


class ChromaIndex:
    """ChromaDB-backed semantic index (used when chromadb is importable)."""

    def __init__(self, path: Path) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._coll = self._client.get_or_create_collection("kernel_library")

    def add(self, id: str, text: str) -> None:
        self._coll.upsert(ids=[id], documents=[text])

    def query(self, text: str, k: int) -> list[str]:
        if self._coll.count() == 0:
            return []
        res = self._coll.query(query_texts=[text], n_results=min(k, self._coll.count()))
        ids = res.get("ids") or [[]]
        return ids[0]


def _make_index(root: Path, prefer_chroma: bool) -> RetrievalIndex:
    if prefer_chroma:
        try:
            return ChromaIndex(root / ".chroma")
        except Exception:
            pass  # missing dep / model — fall back silently
    return LexicalIndex()


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
class Library:
    """File-backed knowledge store with a retrieval index over it."""

    def __init__(self, root: Path | str = "library", *, prefer_chroma: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, LibraryEntry] = {}
        self._index = _make_index(self.root, prefer_chroma)
        self._load_all()

    @property
    def index_kind(self) -> str:
        return type(self._index).__name__

    # ----------------------------------------------------------------- #
    def _load_all(self) -> None:
        for kind in _KINDS:
            for path in (self.root / kind).glob("*.json") if (self.root / kind).exists() else []:
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                entry = LibraryEntry(
                    kind=data.get("kind", kind),
                    title=data.get("title", ""),
                    text=data.get("text", ""),
                    metadata=data.get("metadata", {}),
                )
                self._entries[entry.id] = entry
                self._index.add(entry.id, entry.text)

    def add(self, entry: LibraryEntry) -> str:
        """Add (or idempotently re-add) an entry; returns its id."""
        if entry.kind not in _KINDS:
            entry.kind = "technique"
        path = self.root / entry.kind / f"{entry.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(entry) | {"kind": entry.kind, "id": entry.id}, indent=2))
        if entry.id not in self._entries:
            self._index.add(entry.id, entry.text)
        self._entries[entry.id] = entry
        return entry.id

    def __len__(self) -> int:
        return len(self._entries)

    # ----------------------------------------------------------------- #
    def persist_entries(self, entries: LibraryEntries, *, problem: str, gpu: str = "") -> list[str]:
        """Ingest the structured output of the ``update_library`` subagent."""
        ids: list[str] = []
        base_meta = {"problem": problem, "gpu": gpu}

        for t in entries.techniques:
            title = str(t.get("title", "")).strip() or "technique"
            applies = str(t.get("applies_to", "")).strip()
            detail = str(t.get("detail", "")).strip()
            text = f"[technique] {title}: {detail}" + (f" (applies to: {applies})" if applies else "")
            ids.append(self.add(LibraryEntry("technique", title, text, base_meta | {"applies_to": applies})))

        for f in entries.failed_approaches:
            approach = str(f.get("approach", "")).strip() or "approach"
            why = str(f.get("why_failed", "")).strip()
            text = f"[failed] {approach}: {why}"
            ids.append(self.add(LibraryEntry("failed_approach", approach, text, base_meta)))

        if entries.winning_kernel:
            w = entries.winning_kernel
            approach = str(w.get("approach", "")).strip() or "kernel"
            score = str(w.get("score", "")).strip()
            notes = str(w.get("notes", "")).strip()
            text = f"[winning] {problem} via {approach} ({score}): {notes}"
            ids.append(self.add(LibraryEntry("winning_kernel", f"{problem}:{approach}", text, base_meta | {"score": score})))

        return ids

    def query(self, text: str, k: int = 8) -> list[LibraryEntry]:
        return [self._entries[i] for i in self._index.query(text, k) if i in self._entries]

    def candidates_for(self, brief, k: int = 8) -> list[str]:
        """Retrieval callback for the orchestrator → ``library_retrieval``.

        ``brief`` is a ``ProblemBrief``; we search on its summary + targets and
        return the readable entry strings the subagent reasons over.
        """
        query_text = " ".join(
            [getattr(brief, "summary", ""), getattr(brief, "dtype", "")]
            + list(getattr(brief, "optimization_targets", []) or [])
            + list(getattr(brief, "constraints", []) or [])
        ).strip()
        if not query_text:
            return []
        return [e.text for e in self.query(query_text, k)]
