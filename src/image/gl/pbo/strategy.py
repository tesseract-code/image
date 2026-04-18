from __future__ import annotations

from enum import IntEnum


class PBOBufferingStrategy(IntEnum):
    """
    Number of PBOs cycled during streaming pixel transfers.

    SINGLE (1)
        One PBO.  glMapBuffer stalls the CPU until the GPU finishes reading
        the previous frame.  Only appropriate for non-real-time workloads.

    DOUBLE (2)
        Two PBOs alternated each frame.  While the GPU reads PBO n, the CPU
        writes PBO n+1, hiding transfer latency behind render work.
        Recommended default for most real-time pipelines.

    TRIPLE (3)
        Three PBOs.  Absorbs latency spikes when CPU and GPU workloads do not
        overlap cleanly (e.g. variable-size frames, CPU-side decode jitter).
    """

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3

    @classmethod
    def default(cls) -> PBOBufferingStrategy:
        """Return the recommended strategy for real-time streaming pipelines."""
        return cls.DOUBLE

    @classmethod
    def from_int(cls, value: int) -> PBOBufferingStrategy:
        """Coerce a plain int to a PBOBufferingStrategy member."""
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(str(m.value) for m in cls)
            raise ValueError(
                "Invalid PBOBufferingStrategy value %r. Valid values: %s."
                % (value, valid)
            )

    @property
    def description(self) -> str:
        return _STRATEGY_DESCRIPTIONS[self]


_STRATEGY_DESCRIPTIONS: dict[PBOBufferingStrategy, str] = {
    PBOBufferingStrategy.SINGLE: (
        "Single PBO — simple, may stall CPU on glMapBuffer"
    ),
    PBOBufferingStrategy.DOUBLE: (
        "Double PBO — GPU reads while CPU writes; recommended default"
    ),
    PBOBufferingStrategy.TRIPLE: (
        "Triple PBO — absorbs latency spikes from uneven CPU/GPU overlap"
    ),
}
