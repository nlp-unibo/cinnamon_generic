from pathlib import Path
from typing import Type, Optional, List, Dict, Union, Callable

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Tag, Registration


class ComponentRunnerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='name',
                   type_hint=str,
                   is_required=True)
        config.add(name='tags',
                   type_hint=Tag)
        config.add(name='namespace',
                   type_hint=str,
                   is_required=True)
        config.add(name='run_name',
                   type_hint=str)
        return config


class MultipleComponentRunnerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='registration_keys',
                   type_hint=Union[List[Registration], Callable[[], List[Registration]]],
                   is_required=True)
        config.add(name='runs_names',
                   type_hint=Optional[List[str]])
        config.add(name='run_args',
                   type_hint=Optional[List[Dict]])
        return config


class MultipleRoutineTrainRunnerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='routine_keys',
                   type_hint=Union[List[Registration], Callable[[], List[Registration]]],
                   is_required=True)
        return config


class RoutineInferenceRunnerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='namespace',
                   type_hint=str)
        config.add(name='routine_path',
                   type_hint=Optional[Path])
        config.add(name='run_name',
                   type_hint=str)
        return config
