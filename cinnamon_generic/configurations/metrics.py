from typing import Dict, Any, Set, Callable, List, Optional

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import RegistrationKey

from cinnamon_generic.components.metrics import Metric


class MetricConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='name',
                         type_hint=str,
                         description="Metric name. It is used to uniquely identify the metric",
                         is_required=True)
        config.add_short(name='run_arguments',
                         type_hint=Optional[Dict[str, Any]],
                         description="Additional metric arguments that are required at execution time")
        return config


class LambdaMetricConfig(MetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super(LambdaMetricConfig, cls).get_default()

        config.add_short(name='method',
                         type_hint=Callable,
                         description="Method to invoke for computing metric value",
                         is_required=True)
        config.add_short(name="method_args",
                         type_hint=Optional[Dict[str, Any]],
                         description="Additional method arguments")
        return config
