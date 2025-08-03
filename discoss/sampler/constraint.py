from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any

from lhotse.cut.text import TextExample
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import (
    SamplingConstraint,
)
from lhotse.utils import is_none_or_gt

__all__ = ["TokenConstraint", "token_length_fn"]


def token_length_fn(cut: Cut) -> int:
    """
    A function to compute the number of tokens in a cut.
    This is used by the sampler to determine the length of each cut.
    """
    if isinstance(cut, TextExample):
        assert cut.tokens is not None
        return cut.num_tokens

    # get from feature
    num_tokens = cut.num_frames
    if num_tokens is None:
        # TODO(Feiteng): implement a more accuray way to get num_tokens for Audio & Image & Video Cuts
        # For now, we assume 100 frames per second, which is a common setting.
        num_tokens = int(cut.duration / 0.01)

    return num_tokens


@dataclass
class TokenConstraint(SamplingConstraint):
    """Represents a token-based constraint for sampler classes that sample text data.
    It is defined as maximum total number of tokens in a mini-batch and/or max batch size.

    Similarly to :class:`TimeConstraint`, we support ``quadratic_length`` for quadratic
    token penalty when sampling longer texts.
    """

    max_tokens: int = None
    max_cuts: int = None
    current: int = 0
    num_examples: int = 0
    longest_seen: int = 0
    token_length_fn: Callable[[Cut], int] = token_length_fn
    quadratic_length: Optional[int] = None

    def __post_init__(self) -> None:
        assert is_none_or_gt(self.max_tokens, 0)
        assert is_none_or_gt(self.max_cuts, 0)
        assert is_none_or_gt(self.quadratic_length, 0)

    def add(self, example: TextExample) -> None:
        """Increment the internal token counter for the constraint,
        selecting the right property from the input object.
        """
        if self.max_tokens is not None:
            size = self._maybe_apply_quadratic_correction(self.measure_length(example))
            self.current += size
            self.longest_seen = max(self.longest_seen, size)
        self.num_examples += 1

    def _maybe_apply_quadratic_correction(self, size: int) -> int:
        if self.quadratic_length is None:
            return size
        # For the quadratic complexity case, we add a term that accounts for
        # extra memory occupied by the model. The 1/quadratic_length term causes
        # the effective length to be doubled when it's equal to quadratic_length.
        return int(size + (size**2) / self.quadratic_length)

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_cuts is not None and self.num_examples > self.max_cuts:
            return True
        if self.max_tokens is None:
            return False
        effective_duration = self.num_examples * self.longest_seen
        return effective_duration > self.max_tokens

    def close_to_exceeding(self) -> bool:
        """Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the longest seen cut
        in the current batch, then the batch would have exceeded the constraints.
        """
        if self.max_cuts is not None and self.num_examples >= self.max_cuts:
            return True

        if self.max_tokens is not None:
            effective_size = (self.num_examples + 1) * self.longest_seen
            return effective_size > self.max_tokens
        return False

    def reset(self) -> None:
        """Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_examples = 0
        self.longest_seen = 0

    def measure_length(self, example: TextExample) -> int:
        return self.token_length_fn(example)

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)
