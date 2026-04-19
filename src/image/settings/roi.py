import dataclasses


@dataclasses.dataclass(slots=True, frozen=False)
class ROI:
    """
    Mutable, picklable Region Of Interest.
    Defined in pixel coordinates.
    """
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def as_tuple(self):
        return self.x, self.y, self.width, self.height

