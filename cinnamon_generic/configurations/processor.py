from typing import List

from cinnamon_core.core.registry import RegistrationKey

from cinnamon_generic.components.processor import Processor
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class ProcessorPipelineConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='processors',
                         type_hint=List[RegistrationKey],
                         build_type_hint=List[Processor],
                         description='List of processor configuration registration keys pointing to'
                                     ' process components to be executed in a sequential fashion',
                         is_required=True,
                         is_registration=True)

        return config
