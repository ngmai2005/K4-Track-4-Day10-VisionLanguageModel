"""Runnable Colab pipeline for the Day 10 Qwen3-VL grounding experiments.

Examples (after ``prepare_data.py``):

  python run_lab.py --experiment E0
  python run_lab.py --experiment E1
  python run_lab.py --experiment E4 --rank 4

The script writes transient model outputs under ``work/`` (ignored by git) and
only commits the small, grader-required JSON evidence under ``report/evidence``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import time
from pathlib import Path

import yaml

from lab_helpers import PROMPT
from scripts.labcommon import read_jsonl, write_jsonl
from vl_training import GroundingDataset, make_collate_fn

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROJECTOR_MARKERS = ("visual.merger", "visual.deepstack_merger_list")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(experiment: str) -> dict:
    path = ROOT / "configs" / f"{experiment.lower()}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Essential: E2/E3 must retain trainable full precision projectors.
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def load_base_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config(),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def lora_module_names(model, include_vit: bool) -> list[str]:
    """Return exact module names rather than relying on fragile suffix guesses."""
    import re

    llm = re.compile(
        r"(?:^|\\.)language_model\\.layers\\.\\d+\\."
        r"(?:self_attn\\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\\.(?:gate_proj|up_proj|down_proj))$"
    )
    vit = re.compile(
        r"(?:^|\\.)visual\\.blocks\\.\\d+\\."
        r"(?:attn\\.(?:qkv|proj)|mlp\\.(?:linear_fc1|linear_fc2))$"
    )
    names = [name for name, _ in model.named_modules() if llm.search(name)]
    if include_vit:
        names += [name for name, _ in model.named_modules() if vit.search(name)]
    if not names:
        raise RuntimeError("No LoRA targets matched; print model.named_modules() and update regexes.")
    return names


def apply_setup(model, config: dict):
    """Freeze base weights, install LoRA, then optionally unfreeze projectors."""
    import torch
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    targets = lora_module_names(model, include_vit=config["lora"]["targets"] == "llm+vit")
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora"]["r"]),
            lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    if config.get("train_projector"):
        found = 0
        for name, parameter in peft_model.named_parameters():
            if any(marker in name for marker in PROJECTOR_MARKERS):
                parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
                found += parameter.numel()
        if not found:
            raise RuntimeError("No projector parameters found; do not train a quantized/incorrect model tree.")
    return peft_model, targets


def projector_state_dict(model) -> dict[str, object]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(marker in name for marker in PROJECTOR_MARKERS) and parameter.requires_grad
    }


def restore_projector(model, path: Path) -> None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    missing = set(state) - set(params)
    if missing:
        raise RuntimeError(f"Projector checkpoint incompatible; missing {sorted(missing)[:3]}")
    for name, value in state.items():
        params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def train(experiment: str, config: dict, train_fraction: float = 1.0) -> dict:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(int(config["seed"]))
    model, processor = load_base_model()
    model, targets = apply_setup(model, config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    entries = read_jsonl(ROOT / "data/splits/train.jsonl")
    if train_fraction < 1:
        keep = max(1, round(len(entries) * train_fraction))
        entries = random.Random(int(config["seed"])).sample(entries, keep)
    dataset = GroundingDataset(entries, ROOT / "data", processor)
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True,
        collate_fn=make_collate_fn(processor.tokenizer.pad_token_id), pin_memory=True,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=float(config["lr"]), weight_decay=0.0)
    steps = int(config["max_steps"])
    warmup = max(1, math.ceil(steps * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    accumulation = int(config["grad_accum"])
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}: {loss.item()}")
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if (step + 1) % 25 == 0 or step == 0:
            print(f"{experiment} step={step + 1}/{steps} loss={loss.item() * accumulation:.4f}")

    adapter_dir = ROOT / "work" / experiment.lower() / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    extras = projector_state_dict(model)
    if extras:
        torch.save(extras, adapter_dir / "extra_projector.pt")
    metadata = {
        "trainable_params": trainable_count,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "train_minutes": round((time.monotonic() - started) / 60, 3),
        "matched_lora_modules": len(targets),
        "train_fraction": train_fraction,
    }
    (adapter_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def infer(split: str, out_path: Path, adapter_dir: Path | None = None) -> None:
    """Greedy inference; one image at a time is safest for a free T4."""
    import torch
    from PIL import Image

    model, processor = load_base_model()
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        extra = adapter_dir / "extra_projector.pt"
        if extra.exists():
            restore_projector(model, extra)
    model.eval()
    rows = []
    for index, entry in enumerate(read_jsonl(ROOT / "data/splits" / f"{split}.jsonl"), 1):
        image = Image.open(ROOT / "data" / entry["image_path"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer_ids = generated[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        rows.append({"image_id": entry["image_id"], "raw": raw})
        if index % 25 == 0:
            print(f"inference {index}/{len(rows) if index == 0 else '...'}")
    write_jsonl(out_path, rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def score(experiment: str, pred_path: Path, metadata: dict | None = None) -> None:
    command = [
        "python", "scripts/score.py", "--pred", str(pred_path), "--name", experiment,
        "--out", str(ROOT / "report/evidence" / f"{experiment.lower()}.json"),
    ]
    for key, value in (metadata or {}).items():
        if key in {"trainable_params", "peak_vram_gb", "train_minutes"}:
            command += ["--meta", f"{key}={value}"]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    parser.add_argument("--rank", type=int, choices=[4, 64])
    parser.add_argument("--data-fraction", type=float, choices=[0.1, 1.0])
    args = parser.parse_args()
    experiment = args.experiment
    if experiment == "E0":
        pred = ROOT / "work/e0.jsonl"
        infer("test", pred)
        score("E0", pred)
        return

    config = load_config(experiment if experiment != "E4" else "E1")
    if experiment == "E4":
        # E4 rank/data-size variants use E1 as a controlled baseline by default.
        if args.rank is None and args.data_fraction is None:
            parser.error("E4 requires exactly one of --rank or --data-fraction")
        if args.rank is not None and args.data_fraction is not None:
            parser.error("E4 must change one variable only")
        if args.rank is not None:
            config["lora"]["r"] = args.rank
            config["lora"]["alpha"] = 2 * args.rank
            experiment = f"E4_rank_{args.rank}"
        else:
            experiment = f"E4_data_{args.data_fraction}"
    fraction = args.data_fraction or 1.0
    metadata = train(experiment, config, fraction)
    pred = ROOT / "work" / f"{experiment.lower()}.jsonl"
    infer("test", pred, ROOT / "work" / experiment.lower() / "adapter")
    score(experiment, pred, metadata)


if __name__ == "__main__":
    main()
