from __future__ import annotations

import multiprocessing as mp
from typing import Dict, Any, Optional, Type

from cinnamon_core.core.component import Component
from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.components.calibrator import ValidateCondition, HyperOptCalibrator, GridSearchCalibrator


class UnsetCalibrationConfigurationException(Exception):

    def __init__(self, calibration_config):
        super().__init__(f'Cannot get search space if calibration config is not set! Got {calibration_config}')


class NonTunableConfigurationException(Exception):

    def __init__(self, class_type):
        super().__init__(f'Expected an instance of TunableConfiguration but got {class_type}')


class TunableConfiguration(Configuration):
    """
    The ``TunableConfiguration`` is a ``Configuration`` that supports parameter calibration.
    To allow flexible search space definitions, the ``TunableConfiguration`` leverages a special
    child ``Configuration``: 'calibration_config`.
    This configuration is not bound to any ``Component`` and only has a ``search_space`` parameter.
    The ``search_space`` parameter is a dictionary defining a value search space for each ``TunableConfiguration``
    parameter that has to be calibrated.
    ``TunableConfiguration`` supports nested search spaces (i.e., children that are ``TunableConfiguration`` as well)

    """

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()
        config.add(name='calibration_config',
                   type_hint=RegistrationKey,
                   build_type_hint=Configuration,
                   is_child=True,
                   build_from_registration=False,
                   is_calibration=True,
                   description="Calibration configuration that specifies")
        return config

    def get_search_space(
            self,
            buffer: Optional[Dict[str, Any]] = None,
            parent_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves the search space of the configuration (nesting supported).

        Args:
            buffer: temporary dictionary buffer for recursive function call
            parent_key: optional string suffix to distinguish between configurations parameters

        Returns:
            Flat dictionary containing configuration parameters for which a calibration search space is defined.
        """

        buffer = buffer if buffer is not None else dict()

        # Apply to children as well
        for child_key, child in self.children.items():
            if child.value is None:
                continue

            if isinstance(child.value, RegistrationKey):
                built_child = Registry.build_component_from_key(registration_key=child.value)
            else:
                built_child = child.value

            if isinstance(built_child, Component) and isinstance(built_child.config, TunableConfiguration):
                buffer = built_child.config.get_search_space(buffer=buffer,
                                                             parent_key=f'{parent_key}.{child.name}'
                                                             if parent_key is not None else f'{child.name}')

        # Merge search space
        if self.calibration_config is not None:
            calibration_config_class = Registry.retrieve_configurations_from_key(
                registration_key=self.calibration_config,
                exact_match=True).class_type
            search_space = {f'{parent_key}.{key}' if parent_key is not None else key: value
                            for key, value in calibration_config_class.get_default().search_space.items()}
        else:
            search_space = {}

        buffer = {**buffer, **search_space}

        return buffer


# TODO: add OptunaCalibratorConfig
class CalibratorConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='validator',
                   type_hint=RegistrationKey,
                   description='The component that is run with different hyper-parameter combinations for evaluation',
                   is_required=True,
                   is_child=True)
        config.add(name='validator_args',
                   value={},
                   type_hint=Dict,
                   description='Validator additional run arguments')
        config.add(name='validate_on',
                   value='loss_val_info',
                   type_hint=str,
                   description="metric name to monitor for calibration",
                   is_required=True)
        config.add(name='validate_condition',
                   value=ValidateCondition.MINIMIZATION,
                   type_hint=ValidateCondition,
                   description="whether the ``validate_on`` monitor value should be maximized or minimized",
                   is_required=True)

        return config


class RandomSearchCalibratorConfig(CalibratorConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='tries',
                   value=10,
                   type_hint=int,
                   allowed_range=lambda value: value >= 1,
                   is_required=True,
                   description='Number of hyper-parameter combinations to randomly sample and try')

        return config


class HyperoptCalibratorConfig(CalibratorConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='file_manager_key',
                   type_hint=RegistrationKey,
                   value=RegistrationKey(name='file_manager',
                                         tags={'default'},
                                         namespace='generic'),
                   description="registration info of built FileManager component."
                               " Used for filesystem interfacing")
        config.add(name='max_evaluations',
                   value=-1,
                   type_hint=int,
                   description="number of evaluations to perform for calibration."
                               " -1 allows search space grid search.")
        config.add(name='mongo_directory_name',
                   value='mongodb',
                   description="directory name where mongoDB is located and running",
                   is_required=True)
        config.add(name='mongo_workers_directory_name',
                   value='mongo_workers',
                   description="directory name where mongo workers stored their execution metadata")
        config.add(name='hyperopt_additional_info',
                   type_hint=Optional[Dict[str, Any]],
                   description="additional arguments for hyperopt calibrator")
        config.add(name='use_mongo',
                   value=False,
                   allowed_range=lambda value: value in [False, True],
                   type_hint=bool,
                   description="if enabled, it uses hyperopt mongoDB support for calibration")
        config.add(name='mongo_address',
                   value='localhost',
                   type_hint=str,
                   description="the address of running mongoDB instance")
        config.add(name='mongo_port',
                   value=4000,
                   type_hint=int,
                   description="the port of running mongoDB instance")
        config.add(name='workers',
                   value=2,
                   allowed_range=lambda value: 1 <= value <= mp.cpu_count(),
                   type_hint=int,
                   description="number of mongo workers to run")
        config.add(name='reserve_timeout',
                   value=10.0,
                   type_hint=float,
                   description="Wait time (in seconds) for reserving a calibration "
                               "instance from mongo workers pool")
        config.add(name='max_consecutive_failures',
                   value=2,
                   type_hint=int,
                   description="Maximum number of tentatives before mongo worker is shutdown")
        config.add(name='poll_interval',
                   value=5.0,
                   type_hint=float,
                   description="Wait time for poll request.")
        config.add(name='use_subprocesses',
                   value=False,
                   allowed_range=lambda value: value in [False, True],
                   type_hint=bool,
                   description="If enabled, mongo workers are executed with the"
                               " capability of running subprocesses")
        config.add(name='worker_sleep_interval',
                   value=2.0,
                   type_hint=float,
                   description="Interval time between each mongo worker execution")

        config.add_condition(name='worker_sleep_interval_minimum',
                             condition=lambda parameters: parameters.worker_sleep_interval.value >= 0.5)
        config.add_condition(name="max_evaluations_minimum",
                             condition=lambda parameters: parameters.max_evaluations > 0)

        return config


@register
def register_calibrators():
    Registry.add_and_bind(config_class=Configuration,
                          component_class=GridSearchCalibrator,
                          name='calibrator',
                          tags={'grid'},
                          namespace='generic',
                          is_default=True)
    Registry.add_and_bind(config_class=RandomSearchCalibratorConfig,
                          component_class=RandomSearchCalibratorConfig,
                          name='calibrator',
                          tags={'random'},
                          namespace='generic',
                          is_default=True)
    Registry.add_and_bind(config_class=HyperoptCalibratorConfig,
                          component_class=HyperOptCalibrator,
                          name='calibrator',
                          tags={'hyperopt'},
                          namespace='generic',
                          is_default=True)
