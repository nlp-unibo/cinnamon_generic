from typing import List

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import RegistrationKey

from cinnamon_generic.components.callback import Callback


class CallbackPipelineConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='callbacks',
                         type_hint=List[RegistrationKey],
                         build_type_hint=List[Callback],
                         description='Set of callback configuration registration keys pointing '
                                     'to components to be invoked in sequential format',
                         is_registration=True,
                         is_required=True)

        return config
