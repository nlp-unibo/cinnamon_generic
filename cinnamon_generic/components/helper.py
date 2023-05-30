import random
from typing import Any, Optional

import numpy as np
from cinnamon_core.core.component import Component


class Helper(Component):
    """
    A ``Helper`` is a ``Component`` specialized in handling seeding and backend-specific behaviours.
    This general ``Helper`` component expects an input seed to fix numpy and random packages stochasticity.
    """

    def set_seed(
            self,
            seed: int
    ):
        """
        Sets the seed for reproducibility.

        Args:
            seed: seed to use to fix reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)

    def clear_status(
            self
    ):
        """
        Clears any backed-specific status.
        """

        pass

    def run(
            self,
            seed: Optional[int] = None
    ) -> Any:
        """
        The default behaviour of ``Helper`` is to set the input random seed for reproducibility.

        Args:
            seed: seed to use to fix reproducibility.
        """

        self.set_seed(seed=seed)
