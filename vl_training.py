"""SFT data pipeline for the Day 10 Qwen3-VL grounding lab.

Imports of PyTorch/Transformers are intentionally local so the small geometry
helpers remain usable on a CPU-only development machine.
"""

from __future__ import annotations

from pathlib import Path

from lab_helpers import PROMPT, objects_to_answer


def encode_example(image, prompt: str, answer: str, processor: object) -> dict:
    """Encode one image/chat example and mask every non-assistant token.

    ``prompt_enc`` includes the image placeholders and assistant generation
    prefix.  Its token sequence must be an exact prefix of the complete chat;
    this assertion prevents silently masking the wrong boundary.
    """
    import torch

    user_msg = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    full_msg = user_msg + [{"role": "assistant", "content": answer}]
    prompt_enc = processor.apply_chat_template(
        user_msg, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        full_msg, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    n_prompt = prompt_enc.input_ids.shape[1]
    if not torch.equal(full_enc.input_ids[0, :n_prompt], prompt_enc.input_ids[0]):
        raise ValueError("chat-template prefix mismatch; refusing to build incorrect loss labels")

    labels = full_enc.input_ids.clone()
    labels[:, :n_prompt] = -100
    encoded = {
        "input_ids": full_enc.input_ids[0],
        "attention_mask": full_enc.attention_mask[0],
        "labels": labels[0],
        "pixel_values": full_enc.pixel_values,
        "image_grid_thw": full_enc.image_grid_thw,
    }
    # Current processors can expose this multimodal alignment field.  Preserve
    # it if present and let the collator pad it with text-token semantics.
    if hasattr(full_enc, "mm_token_type_ids"):
        encoded["mm_token_type_ids"] = full_enc.mm_token_type_ids[0]
    return encoded


def make_collate_fn(pad_token_id: int):
    """Return the collator mandated by Qwen3-VL's multimodal representation."""
    import torch

    def collate(batch: list[dict]) -> dict:
        result = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in batch], batch_first=True,
                padding_value=pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in batch], batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [item["labels"] for item in batch], batch_first=True,
                padding_value=-100,
            ),
            # Do not pad vision tensors: their leading dimension contains the
            # flattened patches and is aligned by image_grid_thw.
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        }
        if "mm_token_type_ids" in batch[0]:
            result["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                [item["mm_token_type_ids"] for item in batch], batch_first=True,
                padding_value=0,
            )
        return result

    return collate


class GroundingDataset:
    """Lazy PIL dataset; avoids materializing image pixels in RAM."""

    def __init__(self, entries: list[dict], image_root: str | Path, processor: object, prompt: str = PROMPT):
        self.entries = entries
        self.image_root = Path(image_root)
        self.processor = processor
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.entries[index]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        return encode_example(image, self.prompt, objects_to_answer(row["objects"]), self.processor)
