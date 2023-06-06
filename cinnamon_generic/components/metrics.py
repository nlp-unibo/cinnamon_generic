import abc
from typing import Any, Optional

from cinnamon_core.core.component import Component


class Metric(Component):
    """
    A ``Metric`` is a ``Component`` specialized in computing metrics.
    The ``Metric`` component generally accepts predicted and ground-truth inputs and returns
    the corresponding metric value.
    """

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None,
            as_dict: bool = False
    ) -> Any:
        """
        Computes the metric specified by this component given input predictions (``y_pred``) and
        corresponding ground-truth (``y_true``).

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation
            as_dict: TODO

        Returns:
            The metric result
        """
        pass


class LambdaMetric(Metric):
    """
    A special ``Metric`` component that acts as a wrapper for custom/external metric functions (e.g., scikit-learn).

    """

    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None,
            as_dict: bool = False
    ) -> Any:
        """
        Computes the metric specified by this component given input predictions (``y_pred``) and
        corresponding ground-truth (``y_true``).

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation
            as_dict: TODO

        Returns:
            The metric result
        """

        method_args = self.method_args if self.method_args is not None else {}
        metric_value = self.method(y_pred=y_pred, y_true=y_true, **method_args)
        return metric_value if not as_dict else {self.name: metric_value}
