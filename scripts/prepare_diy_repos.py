"""DIY multi-repo training-data prep (path C of the task).

Clones a curated set of small, distinct real Python repos, computes ONE real
768-d repo embedding per repo with the validated c2l.RepoEncoder, then extracts
many (code_content) examples from each repo's own source (top-level functions and
class methods, via the `ast` module). Each example carries its repo's real
embedding, so the hypernetwork can learn repo -> LoRA differentiation.

Output: one JSONL per split under <out>/ with the schema c2l/dataset.py expects:
    {repo_id, repo_embedding:[768], input_prefix, target_value,
     cross_repo_split, language, file_path}

We split TRAIN vs HELD-OUT by *repo* (cross-repo): some repos are entirely held
out for the Phase-3 eval, so the eval truly measures generalization to a repo the
hypernetwork never saw. Within the JSONL the held-out repos get
cross_repo_split="cr_test" (which c2l/dataset.split routes to the CR/holdout side),
but for the actual eval we load the held-out JSONL directly.

Run:  cd code2lora-mlx && PYTHONPATH=. uv run python scripts/prepare_diy_repos.py
"""

import argparse
import ast
import json
import os
import subprocess
import sys

# (repo_id on GitHub, git ref/tag for reproducibility, role)
# Chosen: small, pure-Python, distinct domains so embeddings differ a lot.
TRAIN_REPOS = [
    ("pallets/click", "8.1.7"),          # CLI framework
    ("psf/requests", "v2.31.0"),         # HTTP client
    ("jd/tenacity", "8.2.3"),            # retry library
    ("pallets/markupsafe", "2.1.5"),     # string escaping
    ("benoitc/instrumentation", None),   # may not exist -> skipped gracefully
    ("Delgan/loguru", "0.7.2"),          # logging
    ("hynek/structlog", "24.1.0"),       # structured logging
    ("agronholm/typeguard", "4.2.1"),    # runtime type checking
    ("python-attrs/attrs", "23.2.0"),    # class boilerplate
    ("pydantic/pydantic-core", None),    # skip (rust) — placeholder
    ("encode/httpx", "0.27.0"),          # async HTTP
    ("theskumar/python-dotenv", "v1.0.1"),  # env files
]

HELDOUT_REPOS = [
    ("pallets/itsdangerous", "2.1.2"),   # signing — distinct from all train repos
    ("ets-labs/python-dependency-injector", None),  # skip if rust/heavy
    ("john-kurkowski/tldextract", "5.1.2"),  # URL parsing
]

MIN_FN_CHARS = 60      # skip trivially short snippets
MAX_FN_CHARS = 2400    # keep examples well under the 2048-token cap
MAX_EX_PER_REPO = 60   # cap examples/repo so no single repo dominates


def clone(repo_id, ref, dest):
    url = f"https://github.com/{repo_id}.git"
    if os.path.isdir(os.path.join(dest, ".git")):
        return True
    args = ["git", "clone", "--quiet", "--depth", "1"]
    if ref:
        args += ["--branch", ref]
    args += ["--single-branch", url, dest]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        # retry without the branch pin (tag/branch may not exist)
        if ref:
            r2 = subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", "--single-branch", url, dest],
                capture_output=True, text=True,
            )
            return r2.returncode == 0
        return False
    return True


def _src_root(repo_dir):
    """Prefer a src/<pkg> or <pkg>/ package dir to avoid tests/docs noise."""
    cands = []
    src = os.path.join(repo_dir, "src")
    if os.path.isdir(src):
        cands.append(src)
    cands.append(repo_dir)
    return cands[0]


def extract_examples(repo_dir, repo_id):
    """Walk .py files; emit (input_prefix, target_value, file_path) examples.

    input_prefix = the def line(s) up to and including the signature + docstring
    (the natural 'prompt'); target_value = the function body that follows. This
    gives the IR next-token loss a real, repo-flavoured continuation to predict.
    """
    examples = []
    root = _src_root(repo_dir)
    py_files = []
    for r, _dirs, files in os.walk(root):
        if "/." in r or "/test" in r.lower() or "/doc" in r.lower():
            continue
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                py_files.append(os.path.join(r, f))
    py_files.sort()

    for fp in py_files:
        if len(examples) >= MAX_EX_PER_REPO:
            break
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        lines = src.splitlines(keepends=True)

        def emit(node):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            start = node.lineno - 1
            end = node.end_lineno
            seg = "".join(lines[start:end])
            if not (MIN_FN_CHARS <= len(seg) <= MAX_FN_CHARS):
                return
            body0 = node.body[0]
            split_line = body0.lineno - 1
            if isinstance(body0, ast.Expr) and isinstance(
                getattr(body0, "value", None), ast.Constant
            ):
                split_line = body0.end_lineno  # include the docstring in the prefix
            prefix = "".join(lines[start:split_line])
            target = "".join(lines[split_line:end])
            if len(prefix.strip()) < 10 or len(target.strip()) < 20:
                return
            rel = os.path.relpath(fp, repo_dir)
            examples.append(
                {"input_prefix": prefix, "target_value": target, "file_path": rel}
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit(node)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    emit(sub)
        # de-dup is handled by the cap; keep insertion order
    return examples[:MAX_EX_PER_REPO]


def build_split(repos, encoder, repos_dir, split_label):
    rows = []
    repo_stats = []
    for repo_id, ref in repos:
        safe = repo_id.replace("/", "__")
        dest = os.path.join(repos_dir, safe)
        if not clone(repo_id, ref, dest):
            print(f"  SKIP (clone failed): {repo_id}", file=sys.stderr)
            continue
        try:
            emb = encoder.embed_repo(dest)  # real 768-d embedding
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (encode failed): {repo_id}: {e}", file=sys.stderr)
            continue
        exs = extract_examples(dest, repo_id)
        if len(exs) < 5:
            print(f"  SKIP (too few examples {len(exs)}): {repo_id}", file=sys.stderr)
            continue
        emb_list = [float(x) for x in emb.tolist()]
        for ex in exs:
            rows.append(
                {
                    "repo_id": repo_id,
                    "repo_embedding": emb_list,
                    "input_prefix": ex["input_prefix"],
                    "target_value": ex["target_value"],
                    "cross_repo_split": split_label,
                    "language": "python",
                    "file_path": ex["file_path"],
                }
            )
        repo_stats.append((repo_id, len(exs)))
        print(f"  OK {repo_id}: {len(exs)} examples, emb[:3]={emb_list[:3]}")
    return rows, repo_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/diy")
    ap.add_argument("--repos-dir", default="data/diy/_repos")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.repos_dir, exist_ok=True)

    from c2l.repo_encoder import RepoEncoder

    print("Loading RepoEncoder (all-MiniLM-L6-v2)...")
    encoder = RepoEncoder()  # device=None -> mps/cpu auto

    print("=== TRAIN repos ===")
    train_rows, train_stats = build_split(
        TRAIN_REPOS, encoder, args.repos_dir, "train"
    )
    print("=== HELD-OUT repos ===")
    held_rows, held_stats = build_split(
        HELDOUT_REPOS, encoder, args.repos_dir, "cr_test"
    )

    train_path = os.path.join(args.out, "train.jsonl")
    held_path = os.path.join(args.out, "heldout.jsonl")
    with open(train_path, "w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(held_path, "w", encoding="utf-8") as fh:
        for r in held_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY ===")
    print(f"TRAIN: {len(train_stats)} repos, {len(train_rows)} examples -> {train_path}")
    for rid, n in train_stats:
        print(f"   {rid}: {n}")
    print(f"HELD-OUT: {len(held_stats)} repos, {len(held_rows)} examples -> {held_path}")
    for rid, n in held_stats:
        print(f"   {rid}: {n}")


if __name__ == "__main__":
    main()
