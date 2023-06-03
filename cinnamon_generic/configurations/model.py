from typing import Type

from cinnamon_core.core.configuration import Configuration, C


class NetworkConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()
        config.add_short(name='epochs',
                         is_required=True,
                         description='Number of epochs to perform to train a model',
                         type_hint=int,
                         allowed_range=lambda epochs: epochs > 0)
        config.add_short(name='stop_training',
                         value=False,
                         description='If enabled, the model stops training immediately',
                         type_hint=bool)

        return config
