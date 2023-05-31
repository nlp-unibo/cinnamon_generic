from typing import Optional, AnyStr, Any

import pytest
from cinnamon_core.core.component import Component
from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import RegistrationKey, Registry
from cinnamon_core.utility.python_utility import get_dict_values_combinations

from cinnamon_generic.configurations.calibrator import TunableConfiguration


class MockComponent(Component):

    def run(
            self,
            serialization_path: Optional[AnyStr] = None,
            serialize: bool = False,
    ) -> Any:
        pass


class ConfigA(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.add_short(name='param1',
                         value=1,
                         type_hint=int)
        config.add_short(name='param2',
                         value=True,
                         type_hint=bool)
        config.add_short(name='child',
                         value=RegistrationKey(name='config_b',
                                               namespace='testing'),
                         is_registration=True)
        config.calibration_config = RegistrationKey(name='calibration',
                                                    tags={'config_a'},
                                                    namespace='testing')

        return config


class ConfigB(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.add_short(name='param1',
                         value=True,
                         type_hint=bool)
        config.calibration_config = RegistrationKey(name='calibration',
                                                    tags={'config_b'},
                                                    namespace='testing')
        return config


class CalConfigA(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.add_short(name='search_space',
                         value={
                             'param1': [1, 2, 3],
                             'param2': [False, True]
                         })
        return config


class CalConfigB(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.add_short(name='search_space',
                         value={
                             'param1': [False, True]
                         })
        return config


@pytest.fixture
def reset_registry():
    Registry.clear()


def test_search_space(
        reset_registry
):
    Registry.register_and_bind(configuration_class=ConfigA,
                               component_class=MockComponent,
                               name='config_a',
                               namespace='testing')
    Registry.register_and_bind(configuration_class=ConfigB,
                               component_class=MockComponent,
                               name='config_b',
                               namespace='testing')

    Registry.register_configuration(configuration_class=CalConfigA,
                                    name='calibration',
                                    tags={'config_a'},
                                    namespace='testing')
    Registry.register_configuration(configuration_class=CalConfigB,
                                    name='calibration',
                                    tags={'config_b'},
                                    namespace='testing')
    config: TunableConfiguration = ConfigA.get_default()
    search_space = config.get_search_space()

    assert len(search_space) == 3
    assert 'param1' in search_space
    assert 'param2' in search_space
    assert 'child.param1' in search_space


def test_search_space_combinations(
        reset_registry
):
    Registry.register_and_bind(configuration_class=ConfigA,
                               component_class=MockComponent,
                               name='config_a',
                               namespace='testing')
    Registry.register_and_bind(configuration_class=ConfigB,
                               component_class=MockComponent,
                               name='config_b',
                               namespace='testing')

    Registry.register_configuration(configuration_class=CalConfigA,
                                    name='calibration',
                                    tags={'config_a'},
                                    namespace='testing')
    Registry.register_configuration(configuration_class=CalConfigB,
                                    name='calibration',
                                    tags={'config_b'},
                                    namespace='testing')
    config: TunableConfiguration = ConfigA.get_default()
    search_space = config.get_search_space()
    combinations = get_dict_values_combinations(search_space)
    assert len(combinations) == 12
