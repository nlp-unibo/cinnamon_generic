import abc
from typing import Any, Dict, List, Optional

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
            y_true: Optional[Any] = None
    ) -> Any:
        """
        Computes the metric specified by this component given input predictions (``y_pred``) and
        corresponding ground-truth (``y_true``).

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation

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
            y_true: Optional[Any] = None
    ) -> Any:
        """
        Computes the metric specified by this component given input predictions (``y_pred``) and
        corresponding ground-truth (``y_true``).

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation

        Returns:
            The metric result
        """

        method_args = self.method_args if self.method_args is not None else {}
        return self.method(y_pred=y_pred,
                           y_true=y_true,
                           **method_args)


class MetricPipeline(Metric):
    """
    A pipeline ``Component`` that runs multiple ``Metric`` components in a sequential fashion.
    """

    def get_metrics(
            self,
            label_name: str
    ) -> List[Metric]:
        """
        Gets the metrics that are specific for the given label.
        If the label name is not found in ``label_metrics_map``, all metrics are returned.

        Args:
            label_name: name of the label to look up in ``label_metrics_map``.

        Returns:
            The list of ``Metric`` components that correspond to the given label name
        """
        if label_name in self.label_metrics_map:
            metric_names = self.label_metrics_map[label_name]
            return [metric for metric in self.metrics if metric.parameters.name in metric_names]
        return self.metrics

    # TODO: generalize the below behaviour: how to handle multi-output and multi-label settings?
    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes all ``Metric`` components for the given input predictions and ground-truth

        Args:
            y_pred: input predictions derived by other components or processes
            y_true: ground-truth for evaluation

        Returns:
            A dictionary where metric names are the keys and metric result are the values
        """

        metric_results = {}

        # Case single output
        if type(y_true) != dict:
            return {metric.name: metric.run(y_pred=y_pred, y_true=y_true) for metric in self.metrics}

        # Case multiple outputs
        for label_name in y_pred:
            label_pred = y_pred[label_name]

            assert label_name in y_true, f'Cannot compute metric for label {label_name}.' \
                                         f' No corresponding ground-truth found.'
            label_true = y_true[label_name]

            label_metrics = self.get_metrics(label_name=label_name)
            for metric in label_metrics:
                metric_result = metric.run(y_pred=label_pred,
                                           y_true=label_true)
                metric_results.setdefault(label_name, {}).setdefault(metric.name, metric_result)

        return metric_results
