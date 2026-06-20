"""Demo on OUR OWN repo (code2lora-mlx): quantitative selftest + a completion
that exercises our own API names."""

from cli import cmd_complete, cmd_selftest

REPO = "."

cmd_selftest(REPO, 12)

prompt = (
    "from c2l import qwen_lora\n\n"
    "# load the frozen Qwen and wrap every q/k/v/o/gate/up/down projection\n"
    "model, tok = qwen_lora.load_base_model()\n"
    "# push the hypernetwork-generated LoRA adapter into every layer\n"
    "qwen_lora."
)
cmd_complete(REPO, prompt, 40)
