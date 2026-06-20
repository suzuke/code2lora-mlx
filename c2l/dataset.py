"""Dataset — faithful MLX port of code2lora-lite/src/dataset.rs.

Mirrors the Candle loader used by the `train` command:
  * load_from_dir walks a directory; .jsonl rows -> RepoPeftBench records;
    .py/.txt files -> CodeExample with a ZERO 768-d repo embedding (exactly
    what dataset.rs does — the encoder is NOT run for the dir-of-source-files
    train path; see dataset.rs:200-211).
  * split(cr_ratio) divides into (CR, IR) by explicit split labels, else by the
    cr_ratio fraction (dataset.rs:271-309).
  * BatchIterator yields (repo_embs: (B, 768) mx.array, code_texts: list[str])
    (dataset.rs:412-432).
"""

import json
import os

import mlx.core as mx
import numpy as np

REPO_EMBED_DIM = 768


class CodeExample:
    __slots__ = (
        "repo_id",
        "repo_embedding",
        "diff_text",
        "code_content",
        "language",
        "split",
        "commit_index",
    )

    def __init__(
        self,
        repo_id: str,
        repo_embedding: np.ndarray,
        code_content: str,
        language: str = "python",
        split: str = "train",
        commit_index=None,
        diff_text=None,
    ):
        self.repo_id = repo_id
        self.repo_embedding = np.asarray(repo_embedding, np.float32).ravel()
        self.code_content = code_content
        self.language = language
        self.split = split
        self.commit_index = commit_index
        self.diff_text = diff_text


# ── JSONL record -> CodeExample, faithful to AssertionRecord.into_example ──
_PREFIX_ALIASES = ("input_prefix", "prefix", "prompt", "question")
_TARGET_ALIASES = ("target_value", "target", "answer", "completion", "assertion")
_EMBED_ALIASES = (
    "repo_embedding",
    "repo_state_embedding",
    "state_embedding",
    "embedding",
)


def _first(record: dict, keys) -> str | None:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _record_to_example(record: dict, row_idx: int) -> CodeExample:
    repo_id = record.get("repo_id") or ""
    repo_id = repo_id if repo_id.strip() else f"unknown_repo_{row_idx}"

    # split: cross_repo_split first, then in_repo_split, then "train".
    cr = record.get("cross_repo_split")
    ir = record.get("in_repo_split")
    split = "train"
    if isinstance(cr, str) and cr.strip():
        split = cr
    elif isinstance(ir, str) and ir.strip():
        split = ir

    code_content = ""
    prefix = _first(record, _PREFIX_ALIASES)
    if prefix:
        code_content += prefix
        if not code_content.endswith("\n"):
            code_content += "\n"
    target = _first(record, _TARGET_ALIASES)
    if target:
        code_content += target
    diff_text = record.get("production_code_diff")
    diff_text = diff_text if (isinstance(diff_text, str) and diff_text.strip()) else None
    if not code_content.strip():
        if diff_text is None:
            raise ValueError(
                "record has no input_prefix/target_value/production_code_diff"
            )
        code_content = diff_text

    # Accept an embedding of ANY length (RepoPeftBench paper embeddings are 2048-d
    # Qwen3-Embedding; our DIY/MiniLM ones are 768-d). We take the embedding as-is
    # so the hypernetwork's repo_embed_dim can be set to match the data. Only when
    # no embedding is present do we zero-fill to the legacy 768-d default.
    emb = None
    for k in _EMBED_ALIASES:
        v = record.get(k)
        if isinstance(v, list) and len(v) > 1:
            emb = np.asarray(v, np.float32)
            break
    if emb is None:
        emb = np.zeros(REPO_EMBED_DIM, np.float32)

    file_path = record.get("file_path") or record.get("test_file")
    language = "python"
    if isinstance(file_path, str) and "." in file_path:
        ext = file_path.rsplit(".", 1)[-1]
        language = "python" if ext == "py" else ext

    return CodeExample(
        repo_id=repo_id,
        repo_embedding=emb,
        code_content=code_content,
        language=language,
        split=split,
        commit_index=record.get("commit_index"),
        diff_text=diff_text,
    )


class CodeDataset:
    def __init__(self, examples: list[CodeExample]):
        self.examples = examples

    @classmethod
    def from_examples(cls, examples: list[CodeExample]) -> "CodeDataset":
        return cls(examples)

    @staticmethod
    def load_jsonl(path: str) -> list[CodeExample]:
        examples = []
        with open(path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                trimmed = line.strip()
                if not trimmed:
                    continue
                record = json.loads(trimmed)
                examples.append(_record_to_example(record, idx))
        return examples

    @classmethod
    def load_from_dir(cls, path: str) -> "CodeDataset":
        """Walk a directory. .jsonl -> records; .py/.txt -> zero-embedding example.

        Faithful to dataset.rs::load_from_dir — for .py/.txt the repo embedding is
        a 768-d ZERO vector (the encoder is not invoked on the dir-train path).
        """
        examples: list[CodeExample] = []
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for fname in sorted(files):
                    fp = os.path.join(root, fname)
                    ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
                    if ext == "jsonl":
                        examples.extend(cls.load_jsonl(fp))
                    elif ext in ("txt", "py"):
                        with open(fp, encoding="utf-8", errors="replace") as f:
                            code = f.read()
                        name = os.path.splitext(fname)[0]
                        language = (
                            "python" if (ext == "py" or "python" in name) else "unknown"
                        )
                        examples.append(
                            CodeExample(
                                repo_id=name,
                                repo_embedding=np.zeros(REPO_EMBED_DIM, np.float32),
                                code_content=code,
                                language=language,
                                split="train",
                            )
                        )
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def is_empty(self) -> bool:
        return len(self.examples) == 0

    def summary(self) -> dict:
        repos = {e.repo_id for e in self.examples}
        langs = {e.language for e in self.examples}
        commit_rows = sum(1 for e in self.examples if e.commit_index is not None)
        return {
            "repo_count": len(repos),
            "language_count": len(langs),
            "commit_rows": commit_rows,
        }

    def split(self, cr_ratio: float):
        """Return (cr_examples, ir_examples). Faithful to dataset.rs::split."""
        n = len(self.examples)
        if n == 0:
            return [], []

        cr_examples, ir_examples = [], []
        has_explicit_split = False
        for ex in self.examples:
            split = ex.split.lower()
            if split != "train":
                has_explicit_split = True
            if (
                split.startswith("cr_")
                or split.startswith("ood_")
                or split in ("test", "val", "validation")
            ):
                cr_examples.append(ex)
            else:
                ir_examples.append(ex)
        if has_explicit_split:
            return cr_examples, ir_examples

        if n == 1:
            return [], list(self.examples)

        cr_count = int(n * cr_ratio)
        cr_count = max(1, min(cr_count, n - 1))
        return self.examples[:cr_count], self.examples[cr_count:]


class BatchIterator:
    """Yields (repo_embs (B,768) mx.array, code_texts list[str]). dataset.rs:412."""

    def __init__(self, examples: list[CodeExample], batch_size: int):
        self.examples = examples
        self.batch_size = max(1, batch_size)

    def __iter__(self):
        pos = 0
        n = len(self.examples)
        while pos < n:
            end = min(pos + self.batch_size, n)
            batch = self.examples[pos:end]
            pos = end
            embs = np.stack([ex.repo_embedding for ex in batch], axis=0)
            repo_embs = mx.array(embs)
            texts = [ex.code_content for ex in batch]
            yield repo_embs, texts
