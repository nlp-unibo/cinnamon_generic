import abc
from typing import Any, Optional

from cinnamon_core.core.component import Component
from cinnamon_generic.components.pipeline import Pipeline


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
            as_dict: whether metric results are reported in dict format or not

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
            as_dict: whether metric results are reported in dict format or not

        Returns:
            The metric result
        """

        method_args = self.method_args if self.method_args is not None else {}
        metric_value = self.method(y_pred=y_pred, y_true=y_true, **method_args)
        return metric_value if not as_dict else {self.name: metric_value}


class MetricPipeline(Pipeline, Metric):

    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None,
            as_dict: bool = False
    ) -> Any:
        """
        Executes each child metric.run() in the pipeline.

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation
            as_dict: whether metric results are reported in dict format or not

        Returns:
            A list or dict containing all metrics results
        """

        metrics = self.get_pipeline()
        result = []
        for metric in metrics:
            metric_result = metric.run(y_pred=y_pred,
                                       y_true=y_true,
                                       as_dict=as_dict)
            result.append(metric_result)

        if as_dict:
            return {key: value for metric_dict in result for key, value in metric_dict.items()}

        return result
