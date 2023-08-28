from typing import Optional, Union, List

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import RegistrationKey
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.data_splitter import TTSplitter
from cinnamon_generic.components.helper import Helper
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class RoutineConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config: Configuration = super().get_default()

        config.add(name='seeds',
                   type_hint=Union[List[int], int],
                   description="Seeds to use for reproducible benchmarks",
                   is_required=True)
        config.add(name='data_loader',
                   type_hint=RegistrationKey,
                   build_type_hint=DataLoader,
                   description="DataLoader component used to for data loading",
                   is_required=True,
                   is_child=True)
        config.add(name='data_splitter',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=TTSplitter,
                   description="Data splitter component for creating train/val/test splits",
                   is_required=True,
                   is_child=True)
        config.add(name='pre_processor',
                   type_hint=RegistrationKey,
                   build_type_hint=Processor,
                   description="Processor component used to for data pre-processing",
                   is_required=True,
                   is_child=True,
                   build_from_registration=False)
        config.add(name='post_processor',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Optional[Processor],
                   description="Processor component used to for data post-processing",
                   is_child=True,
                   build_from_registration=False)
        config.add(name='model_processor',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Optional[Processor],
                   description="Processor component used to for model output post-processing",
                   is_child=True,
                   build_from_registration=False)
        config.add(name='routine_processor',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Optional[Processor],
                   description="Processor component used to for routine results processing",
                   is_child=True,
                   build_from_registration=True)
        config.add(name='model',
                   type_hint=RegistrationKey,
                   build_type_hint=Model,
                   description="Model component used to wrap a machine learning model ",
                   is_required=True,
                   is_child=True,
                   build_from_registration=False)
        config.add(name='callbacks',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Callback,
                   description="Callback component for customized control flow and side effects",
                   is_child=True,
                   build_from_registration=False)
        config.add(name='metrics',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Metric,
                   description="Metric component for routine evaluation",
                   is_child=True,
                   build_from_registration=False)
        config.add(name='helper',
                   type_hint=Optional[RegistrationKey],
                   build_type_hint=Optional[Helper],
                   description="Helper component for reproducibility and backend management",
                   is_child=True)

        return config


class CVRoutineConfig(RoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add(name='max_folds',
                   value=-1,
                   type_hint=int,
                   description='Number of folds to consider. '
                               'If -1, all folds provided by the data splitter will be considered.')
        return config