from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class DataLoaderConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='name',
                   type_hint=str,
                   description="Unique dataset identifier",
                   is_required=True)
        config.add(name='has_test_split_only',
                   value=False,
                   type_hint=bool,
                   description="Whether the ``DataLoader`` has test split only or not")
        config.add(name='has_val_split',
                   value=True,
                   type_hint=bool,
                   description="Whether the ``DataLoader`` has a val split or not")
        config.add(name='has_test_split',
                   value=True,
                   type_hint=bool,
                   description="Whether the ``DataLoader`` has a test split or not")

        return config
