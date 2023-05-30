from pathlib import Path
from typing import Any, AnyStr, Optional, Union, List

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import RegistrationKey, register, Registry
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager
from cinnamon_generic.components.helper import Helper
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.components.routine import PrebuiltCVSplitter, PrebuiltLOOSplitter, CVSplitter, LOOSplitter
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class RoutineConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config: Configuration = super().get_default()

        config.add_short(name='seeds',
                         type_hint=Union[List[int], int],
                         description="Seeds to use for reproducible benchmarks",
                         is_required=True)
        config.add_short(name='validation_percentage',
                         type_hint=Optional[float],
                         description='Training set percentage to use as validation split')
        config.add_short(name='data_loader',
                         type_hint=RegistrationKey,
                         build_type_hint=DataLoader,
                         description="DataLoader component used to for data loading",
                         is_required=True,
                         is_registration=True)
        config.add_short(name='pre_processor',
                         type_hint=RegistrationKey,
                         build_type_hint=Processor,
                         description="Processor component used to for data pre-processing",
                         is_required=True,
                         is_registration=True,
                         build_from_registration=False)
        config.add_short(name='post_processor',
                         type_hint=Optional[RegistrationKey],
                         build_type_hint=Optional[Processor],
                         description="Processor component used to for data post-processing",
                         is_registration=True,
                         build_from_registration=False)
        config.add_short(name='routine_processor',
                         type_hint=Optional[RegistrationKey],
                         build_type_hint=Optional[Processor],
                         description="Processor component used to for routine results processing",
                         is_registration=True,
                         build_from_registration=True)
        config.add_short(name='model',
                         type_hint=RegistrationKey,
                         build_type_hint=Model,
                         description="Model component used to wrap a machine learning model ",
                         is_required=True,
                         is_registration=True,
                         build_from_registration=False)
        config.add_short(name='callbacks',
                         type_hint=Optional[RegistrationKey],
                         build_type_hint=Callback,
                         description="Callback component for customized control flow and side effects",
                         is_registration=True,
                         build_from_registration=False)
        config.add_short(name='metrics',
                         type_hint=Optional[RegistrationKey],
                         build_type_hint=Metric,
                         description="Metric component for routine evaluation",
                         is_registration=True,
                         build_from_registration=False)
        config.add_short(name='helper',
                         type_hint=Optional[RegistrationKey],
                         build_type_hint=Optional[Helper],
                         description="Helper component for reproducibility and backend management",
                         is_registration=True)

        return config


class TrainAndTestRoutineConfig(RoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config: Configuration = super(TrainAndTestRoutineConfig, cls).get_default()

        config.add_short(name='test_percentage',
                         type_hint=Optional[float],
                         description='Training set percentage to use as test split.')
        config.add_short(name='has_val_split',
                         value=True,
                         type_hint=bool,
                         description='If true, val data is considered as a data split. '
                                     'If no val data is provided, it is built via random split')
        config.add_short(name='has_test_split',
                         value=True,
                         type_hint=bool,
                         description="If true, test data is distinct from training data "
                                     "and no data split is required")
        return config


class CVRoutineConfig(RoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config: Configuration = super(CVRoutineConfig, cls).get_default()

        config.add_short(name='file_manager_registration_key',
                         value=RegistrationKey(name='file_manager',
                                               tags={'default'},
                                               namespace='generic'),
                         type_hint=RegistrationKey,
                         build_type_hint=FileManager,
                         description="the FileManager component used for filesystem interfacing",
                         is_required=True)
        config.add_short(name='cv_splitter',
                         type_hint=RegistrationKey,
                         build_type_hint=CVSplitter,
                         description="The splitter component for cross-validation folds computation",
                         is_required=True)
        config.add_short(name='split_key',
                         type_hint=Any,
                         description="key for splitting data",
                         is_required=True)
        config.add_short(name='held_out_key',
                         value='validation',
                         allowed_range=lambda value: value in ['validation', 'test'],
                         type_hint=str,
                         description="Which data split key (e.g., test, validation) built folds belong to")

        return config


class LOORoutineConfig(RoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config: Configuration = super(LOORoutineConfig, cls).get_default()

        config.add_short(name='file_manager_registration_key',
                         value=RegistrationKey(name='file_manager',
                                               tags={'default'},
                                               namespace='generic'),
                         type_hint=RegistrationKey,
                         build_type_hint=FileManager,
                         description="the FileManager component used for filesystem interfacing",
                         is_required=True)
        config.add_short(name='cv_splitter',
                         type_hint=RegistrationKey,
                         build_type_hint=LOOSplitter,
                         description="The splitter component for cross-validation folds computation",
                         is_required=True)
        config.add_short(name='split_key',
                         type_hint=Any,
                         description="key for splitting data",
                         is_required=True)

        return config


class PrebuiltSplitterConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='prebuilt_filename',
                         type_hint=str,
                         description="Filename for storing/loading pre-built folds",
                         is_required=True)
        config.add_short(name='prebuilt_folder',
                         value='prebuilt_folds',
                         type_hint=Union[AnyStr, Path],
                         description="Folder name where to store pre-built fold files",
                         is_required=True)
        config.add_short(name='file_manager_registration_key',
                         value=RegistrationKey(name='file_manager',
                                               tags={'default'},
                                               namespace='generic'),
                         description="Registration key pointing to built FileManager component",
                         is_required=True)

        return config


class PrebuiltCVSplitterConfig(PrebuiltSplitterConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='n_splits',
                         type_hint=int,
                         description="Number of splits to perform to build folds",
                         is_required=True)
        config.add_short(name='shuffle',
                         type_hint=bool,
                         value=True,
                         description="If True, input data is shuffled")
        config.add_short(name='cv_type',
                         value=KFold,
                         type_hint=_BaseKFold,
                         description="Internal cross-validation fold builder",
                         is_required=True)

        return config


@register
def register_routine_splitters():
    Registry.register_and_bind(configuration_class=PrebuiltCVSplitterConfig,
                               component_class=PrebuiltCVSplitter,
                               name='routine_splitter',
                               tags={'prebuilt', 'cv'},
                               namespace='generic')

    Registry.register_and_bind(configuration_class=PrebuiltSplitterConfig,
                               component_class=PrebuiltLOOSplitter,
                               name='routine_splitter',
                               tags={'prebuilt', 'loo'},
                               namespace='generic')
