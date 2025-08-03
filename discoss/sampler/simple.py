import warnings
from typing import Any, Dict, Optional

from lhotse import CutSet
from lhotse.dataset.sampling import SimpleCutSampler as LhotseSimpleCutSampler

from .constraint import TokenConstraint

__all__ = ["SimpleCutSampler"]


class SimpleCutSampler(LhotseSimpleCutSampler):
    """A simple cut sampler that inherits from Lhotse's SimpleCutSampler."""

    def __init__(
        self,
        cuts: CutSet,
        constraint: TokenConstraint,
        shuffle: bool = False,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(
            cuts=cuts,
            max_cuts=1000000000,  # No limit on the number of cuts
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.time_constraint = constraint

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "token_constraint": self.time_constraint.state_dict(),
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        time_constraint = TokenConstraint(**state_dict.pop("token_constraint"))
        if self.time_constraint != time_constraint:
            warnings.warn(
                "SimpleCutSampler.load_state_dict(): Inconsistent token_constraint:\n"
                f"expected {self.time_constraint}\n"
                f"received {time_constraint}\n"
                f"We will overwrite the settings with the received state_dict."
            )
        self.time_constraint = time_constraint

        super().load_state_dict(state_dict)
