from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from megatron.energon.flavors.base_dataset import Sample
import torch


@dataclass
class ImageConvSample(Sample):
    """Sample type for Image Conv dataset."""
    
    conversations: List[Dict]
    images: List[torch.Tensor]
    image_count: int
    image_sizes: List[Tuple[int, int]]
    _source: Optional[str] = None


"""
def sample_loader(sample: dict) -> dict:
    metadata = sample["metadata.json"]
    image_count = metadata["image_count"]
    images = [sample.get(f"image_{i}.jpg") for i in range(image_count)]
    image_sizes = metadata["size"]  # [] or [H, W] or [[H, W], [H, W], ...]
    if not isinstance(image_sizes, list):
        image_sizes = [image_sizes]
    elif len(image_sizes) > 0 and not isinstance(image_sizes[0], list):
        image_sizes = [image_sizes]
    elif len(image_sizes) < 1:
        image_sizes = []
        image_count = 0

    conversations = sample["conversations.json"]
    return dict(
        __key__=sample["__key__"],
        __restore_key__=sample["__restore_key__"],
        conversations=conversations,
        images=images,
        image_count=image_count,
        image_sizes=image_sizes,
        _source=metadata.get("source")
    )


def part_filter(part: str) -> bool:
    return True
"""