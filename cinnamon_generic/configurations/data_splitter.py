from pathlib import Path
from typing import Any, AnyStr, Union, Type, Hashable, Optional

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, register, Registry
from cinnamon_generic.components.data_splitter import InternalTTSplitter, SklearnTTSplitter, TTSplitter
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class TTSplitterConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='validation_size',
                   type_hint=Optional[float],
                   description='Training set percentage to use as validation split')
        config.add(name='test_size',
                   type_hint=Optional[float],
                   description='Training set percentage to use as test split')
        config.add(name='splitter_type',
                   type_hint=InternalTTSplitter,
                   description='Splitter class for performing data split',
                   is_required=True)
        config.add(name='splitter_args',
                   value={},
                   description="Arguments for creating a splitter instance")

        return config


class SklearnTTSplitterConfig(TTSplitterConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.splitter_type = SklearnTTSplitter
        config.splitter_args = {
            'random_state': 42,
            'shuffle': True
        }
        return config


class CVSplitterConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='splitter_type',
                   value=KFold,
                   type_hint=_BaseKFold,
                   description='Splitter class for performing data split',
                   is_required=True)
        config.add(name='splitter_args',
                   value={
                       'n_splits': 5,
                       'shuffle': True

                   },
                   description="Arguments for creating a splitter instance")
        config.add(name='X_key',
                   type_hint=Hashable,
                   description='Column name for input data')
        config.add(name='y_key',
                   type_hint=Any,
                   description="Column name for output data",
                   is_required=True)
        config.add(name='group_key',
                   type_hint=Hashable,
                   description='Column name for grouping')
        config.add(name='held_out_key',
                   value='validation',
                   allowed_range=lambda value: value in ['validation', 'test'],
                   type_hint=str,
                   description="Which data split key (e.g., test, validation) built folds belong to")
        config.add(name='validation_n_splits',
                   value=config.splitter_args['n_splits'],
                   type_hint=int,
                   description="Number of splits to perform to build folds",
                   is_required=True)
        return config


class PrebuiltCVSplitterConfig(CVSplitterConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add(name='prebuilt_filename',
                   type_hint=str,
                   description="Filename for storing/loading pre-built folds",
                   is_required=True)
        config.add(name='prebuilt_folder_name',
                   value='prebuilt_folds',
                   type_hint=Union[AnyStr, Path],
                   description="Folder name where to store pre-built fold files",
                   is_required=True)
        config.add(name='file_manager_key',
                   value=RegistrationKey(name='file_manager',
                                         tags={'default'},
                                         namespace='generic'),
                   description="Registration key pointing to built FileManager component",
                   is_required=True)

        return config


@register
def register_data_splitter_configurations():
    Registry.add_and_bind(config_class=SklearnTTSplitterConfig,
                          component_class=TTSplitter,
                          name='data_splitter',
                          tags={'tt', 'sklearn'},
                          namespace='generic')
