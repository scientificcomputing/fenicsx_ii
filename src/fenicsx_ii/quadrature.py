from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Quadrature:
    name: str
    points: npt.NDArray[np.floating]
    weights: npt.NDArray[np.floating]
    scales: npt.NDArray[np.floating]
