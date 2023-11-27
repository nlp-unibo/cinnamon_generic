from typing import Callable

from cinnamon_core.core.component import Component
from cinnamon_core.core.configuration import C


# TODO: update runners
class ComponentRunner(Component):

    def parse_config(
            self,
            config: C
    ) -> C:
        return config

    def run(
            self,
    ):
        return self.parse_config(self.config)


class MultipleComponentRunner(ComponentRunner):

    def parse_config(
            self,
            config: C
    ) -> C:
        if isinstance(config.registration_keys, Callable):
            config.registration_keys = config.registration_keys()
        return config


class MultipleRoutineTrainRunner(ComponentRunner):

    def parse_config(
            self,
            config: C
    ) -> C:
        if isinstance(config.routine_keys, Callable):
            config.routine_keys = config.routine_keys()
        return config
