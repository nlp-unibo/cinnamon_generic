from argparse import ArgumentParser
from typing import Callable

from cinnamon_core.core.component import Component
from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, RegistrationKey, NotRegisteredException
from cinnamon_core.utility import logging_utility


class ComponentRunner(Component):

    def parse_config(
            self,
            config: C
    ) -> C:
        return config

    def run(
            self,
    ):
        parser = ArgumentParser()
        parser.add_argument('--name', '-n', nargs='?', const=None, type=str)
        parser.add_argument('--tags', '-t', nargs='+', const=None, type=str)
        parser.add_argument('--namespace', '-ns', nargs='?', const=None, type=str)
        args = parser.parse_args()

        replace_key = RegistrationKey(name=args.name,
                                      tags=set(args.tags) if args.tags is not None else args.tags,
                                      namespace=args.namespace)
        try:
            replace_config = Registry.build_configuration_from_key(registration_key=replace_key)
            return self.parse_config(replace_config)
        except NotRegisteredException:
            logging_utility.logger.warning(f'Could not find registration key {replace_key}. '
                                           f'The runner will ignore the submitted registration key.')
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
