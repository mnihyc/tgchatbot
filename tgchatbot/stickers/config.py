"""Sticker retrieval controls in the existing application environment."""
from dataclasses import dataclass
import math
from tgchatbot.operational import from_env


@dataclass(frozen=True)
class StickerConfig:
    candidate_count: int = 5
    max_candidates: int = 8
    retrieval_depth: int = 64
    recent_deliveries: int = 20
    cached_sessions: int = 32
    near_duplicate_similarity: float = 0.985
    intensity_rank_weight: float = 0.5

    def __post_init__(self):
        for name in ('candidate_count', 'max_candidates', 'retrieval_depth', 'recent_deliveries', 'cached_sessions'):
            if getattr(self, name) < 1:
                raise ValueError(f'STICKER_{name.upper()} must be positive')
        if not -1 <= self.near_duplicate_similarity <= 1:
            raise ValueError('STICKER_NEAR_DUPLICATE_SIMILARITY must be a cosine similarity between -1 and 1')
        if not math.isfinite(self.intensity_rank_weight) or self.intensity_rank_weight < 0:
            raise ValueError('STICKER_INTENSITY_RANK_WEIGHT must be finite and nonnegative')

    @classmethod
    def from_env(cls):
        return from_env(cls, 'STICKER')
