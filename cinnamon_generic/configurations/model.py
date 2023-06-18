from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class NetworkConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()
        config.add(name='epochs',
                   is_required=True,
                   description='Number of epochs to perform to train a model',
                   type_hint=int,
                   allowed_range=lambda epochs: epochs > 0)
        config.add(name='stop_training',
                   value=False,
                   description='If enabled, the model stops training immediately',
                   type_hint=bool)

        return config
