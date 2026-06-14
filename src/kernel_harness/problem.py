"""Fetch GPU MODE problem specs from the ``gpu-mode/reference-kernels`` repo.

The leaderboard's numeric id (e.g. ``543``) only appears in the gpumode.com web
URL. The canonical identifier everywhere else — in the reference repo and in
``popcorn submit --leaderboard <name>`` — is the problem *name* (e.g.
``vectoradd_v2``). So the harness keys on the name.

Each problem set is a ``problems/<set>.yaml`` file mapping problem names to a
directory plus the leaderboard GPUs. Each problem directory contains:

* ``reference.py`` — the PyTorch reference implementation (ground truth)
* ``task.py``      — the input/output schema and harness glue
* ``task.yml``     — problem metadata (description, tolerances, test/benchmark sizes)
* ``submission.py``— a template submission

We download the three spec files and hand them, unparsed, to the
problem-understander subagent. See CONTEXT.md → "Entry Point".
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field

import httpx
import yaml

_API_BASE = "https://api.github.com/repos/gpu-mode/reference-kernels/contents"
_RAW_BASE = "https://raw.githubusercontent.com/gpu-mode/reference-kernels/main"
_BRANCH = "main"

# Spec files we pull from each problem directory.
_SPEC_FILES = ("task.yml", "task.py", "reference.py")


class ProblemNotFoundError(LookupError):
    """Raised when a leaderboard name matches no problem in any set."""


@dataclass
class Problem:
    """A fully-resolved problem ready to feed into the loop.

    Carries two views of the problem's files:

    * ``spec_files`` — human/LLM-facing docs (task.yml, task.py, reference.py)
      handed to the problem-understander subagent as prompt context.
    * ``stage_files`` — everything the local eval harness needs on disk
      (eval.py, utils.py, task.py, reference.py), keyed by their staged
      filename, resolved from the ``files:`` list in task.yml. The submission
      itself is excluded (it's written per-candidate at run time).

    Plus the parsed ``tests`` / ``benchmarks`` case lists used to materialize the
    eval harness's test-spec file. See ``evalproto`` for how these are used.
    """

    name: str                 # canonical popcorn leaderboard name, e.g. "vectoradd_v2"
    problem_set: str          # set file stem, e.g. "pmpp_v2"
    directory: str            # repo path, e.g. "pmpp_v2/vectoradd_py"
    gpus: list[str]           # leaderboard target GPUs, e.g. ["B200", "H100", ...]
    spec_files: dict[str, str]  # filename -> raw text (task.yml, task.py, reference.py)

    # Execution staging (populated from task.yml).
    description: str = ""
    entry_point: str = "eval.py"            # task.yml config.main
    submission_filename: str = "submission.py"
    stage_files: dict[str, str] = field(default_factory=dict)  # staged name -> content
    tests: list[dict] = field(default_factory=list)            # correctness cases
    benchmarks: list[dict] = field(default_factory=list)       # timing cases
    timeouts: dict[str, int] = field(default_factory=dict)     # test/benchmark/ranked

    def default_gpu(self) -> str:
        """First listed leaderboard GPU, used as the submission default."""
        return self.gpus[0] if self.gpus else ""

    def as_prompt_context(self) -> str:
        """Render the spec files as a single block for a subagent prompt."""
        parts = [
            f"# Problem: {self.name}  (set: {self.problem_set})",
            f"# Leaderboard GPUs: {', '.join(self.gpus) or 'unknown'}",
            f"# Repo directory: {self.directory}",
        ]
        for fname, text in self.spec_files.items():
            parts.append(f"\n===== {fname} =====\n{text}")
        return "\n".join(parts)


class ProblemFetcher:
    """Resolves leaderboard names against the reference-kernels repo."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30, follow_redirects=True)
        self._owns_client = client is None
        self._raw_cache: dict[str, str] = {}

    def __enter__(self) -> "ProblemFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def fetch(self, leaderboard: str) -> Problem:
        """Resolve a leaderboard name to a fully-populated ``Problem``.

        Raises ``ProblemNotFoundError`` if no set lists the name.
        """
        entry = self._find_problem_entry(leaderboard)
        directory = entry["directory"]
        spec_files = self._fetch_spec_files(directory)

        task_yaml = yaml.safe_load(spec_files.get("task.yml", "") or "") or {}
        stage_files, submission_name = self._resolve_stage_files(directory, task_yaml)

        return Problem(
            name=entry["name"],
            problem_set=entry["_set"],
            directory=directory,
            gpus=list(entry.get("gpus") or []),
            spec_files=spec_files,
            description=str(task_yaml.get("description", "")).strip(),
            entry_point=str((task_yaml.get("config") or {}).get("main", "eval.py")),
            submission_filename=submission_name,
            stage_files=stage_files,
            tests=list(task_yaml.get("tests") or []),
            benchmarks=list(task_yaml.get("benchmarks") or []),
            timeouts={
                k: int(task_yaml[k])
                for k in ("test_timeout", "benchmark_timeout", "ranked_timeout")
                if k in task_yaml
            },
        )

    def _resolve_stage_files(
        self, directory: str, task_yaml: dict
    ) -> tuple[dict[str, str], str]:
        """Fetch every file the eval harness needs on disk, per task.yml ``files``.

        Returns ``(name -> content, submission_filename)``. The submission entry
        (``source == "@SUBMISSION@"``) is recorded by name but not fetched — its
        content is the candidate kernel, written per-run by ``evalproto``.
        """
        stage_files: dict[str, str] = {}
        submission_name = "submission.py"
        for entry in task_yaml.get("files") or []:
            name = entry.get("name")
            source = entry.get("source")
            if not name or not source:
                continue
            if source == "@SUBMISSION@":
                submission_name = name
                continue
            # Resolve sources like "../utils.py" relative to the problem directory.
            resolved = posixpath.normpath(posixpath.join(directory, source))
            stage_files[name] = self._fetch_raw(resolved)
        return stage_files, submission_name

    def list_problems(self) -> list[str]:
        """Return every known leaderboard name across all problem sets."""
        names: list[str] = []
        for set_path in self._list_problem_set_files():
            doc = self._load_yaml(set_path)
            for prob in doc.get("problems") or []:
                if name := prob.get("name"):
                    names.append(name)
        return sorted(names)

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #
    def _find_problem_entry(self, leaderboard: str) -> dict:
        """Search every set YAML for a problem with ``name == leaderboard``."""
        for set_path in self._list_problem_set_files():
            doc = self._load_yaml(set_path)
            set_stem = set_path.rsplit("/", 1)[-1].removesuffix(".yaml")
            for prob in doc.get("problems") or []:
                if prob.get("name") == leaderboard:
                    return {**prob, "_set": set_stem}
        raise ProblemNotFoundError(
            f"no problem named {leaderboard!r} found in gpu-mode/reference-kernels. "
            "Use the canonical leaderboard name (e.g. 'vectoradd_v2'); "
            "run with --list-problems to see all names."
        )

    def _list_problem_set_files(self) -> list[str]:
        """List problem-set YAML names via the GitHub contents API.

        Returns paths relative to the ``problems/`` directory (e.g. ``amd.yaml``),
        matching the convention ``_fetch_raw`` expects.
        """
        resp = self._client.get(f"{_API_BASE}/problems", params={"ref": _BRANCH})
        resp.raise_for_status()
        return [
            item["name"]
            for item in resp.json()
            if item["type"] == "file" and item["name"].endswith(".yaml")
        ]

    def _load_yaml(self, repo_path: str) -> dict:
        """Fetch and parse a YAML file from the repo by path."""
        text = self._fetch_raw(repo_path)
        return yaml.safe_load(text) or {}

    def _fetch_spec_files(self, directory: str) -> dict[str, str]:
        """Download the known spec files from a problem directory.

        Missing files are skipped (not every problem ships every spec file).
        """
        out: dict[str, str] = {}
        for fname in _SPEC_FILES:
            try:
                out[fname] = self._fetch_raw(f"{directory}/{fname}")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    continue
                raise
        return out

    def _fetch_raw(self, repo_path: str) -> str:
        """Fetch a file's raw text from the repo's main branch (cached per fetcher)."""
        if repo_path in self._raw_cache:
            return self._raw_cache[repo_path]
        resp = self._client.get(f"{_RAW_BASE}/problems/{repo_path}")
        resp.raise_for_status()
        self._raw_cache[repo_path] = resp.text
        return resp.text
