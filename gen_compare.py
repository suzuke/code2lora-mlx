"""Quick base-vs-adapted greedy completion on a held-out repo, using the TAIL of
a real (long) prefix as the prompt (the RepoPeftBench prefixes are 600-2800 chars,
so eval_heldout.py's 20-400 char filter matched none). For the report's quoted
completions only."""
import argparse, json
import mlx.core as mx
from mlx.utils import tree_unflatten
from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.hypernetwork import Code2LoRAHead
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--repo-embed-dim", type=int, default=2048)
ap.add_argument("--tail", type=int, default=320)
ap.add_argument("--gen-tokens", type=int, default=32)
ap.add_argument("--n", type=int, default=2)
args = ap.parse_args()

cfg = HypernetworkConfig(repo_embed_dim=args.repo_embed_dim, max_lora_scale=1.0)
rows = [json.loads(l) for l in open(args.heldout) if l.strip()]
rows = [r for r in rows if r["repo_id"] == args.repo][: args.n]
emb = np.asarray(rows[0]["repo_embedding"], np.float32)

model, tok = qwen_lora.load_base_model()
model.set_dtype(mx.float32); model.freeze()
hn = Code2LoRAHead(cfg)
hn.update(tree_unflatten(list(mx.load(args.ckpt).items()))); mx.eval(hn.parameters())
all_lora = hn.forward_all(mx.array(emb[None, :]))

for i, r in enumerate(rows):
    prompt = r["input_prefix"].rstrip()[-args.tail:]
    qwen_lora.clear_lora(model)
    base = qwen_lora.generate_text(model, tok, prompt, args.gen_tokens)
    qwen_lora.clear_lora(model); qwen_lora.inject_lora(model, all_lora)
    adapt = qwen_lora.generate_text(model, tok, prompt, args.gen_tokens)
    qwen_lora.clear_lora(model)
    print(f"\n===== Prompt {i+1} (tail) =====\n...{prompt[-160:]!r}")
    print(f"\n[BASE]   {base!r}")
    print(f"[ADAPT]  {adapt!r}")
