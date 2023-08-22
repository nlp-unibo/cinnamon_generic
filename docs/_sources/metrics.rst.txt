.. _metrics:

Metrics
*************************************

A ``Metric`` is a ``Component`` specialized in computing metrics.

The ``Metric`` component generally accepts predicted and ground-truth inputs and returns the corresponding metric value.

.. code-block:: python

    value = metric.run(y_pred=y_pred, y_true=y_true)

The ``Metric`` uses the ``MetricConfig`` as the default configuration template:

.. code-block:: python

    class MetricConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='name',
                       type_hint=str,
                       description="Metric name. It is used to uniquely identify the metric",
                       is_required=True)

            config.add(name='run_arguments',
                       type_hint=Optional[Dict[str, Any]],
                       description="Additional metric arguments that are required at execution time")
            return config

Cinnamon provides the following ``Metric`` components:

- ``LambdaMetric``: wraps custom metric functions
- ``MetricPipeline``: a ``Pipeline`` component specialized for metrics.


-------------------------------
``LambdaMetric``
-------------------------------

A special ``Metric`` component that acts as a wrapper for custom/external metric functions (e.g., scikit-learn).

The ``LambdaMetric`` uses ``LambdaMetricConfig`` as the default configuration template:

.. code-block:: python

    class LambdaMetricConfig(MetricConfig):

        @classmethod
        def get_default(
                cls
        ):
            config = super(LambdaMetricConfig, cls).get_default()

            config.add(name='method',
                       type_hint=Callable,
                       description="Method to invoke for computing metric value",
                       is_required=True)

            config.add(name="method_args",
                       type_hint=Optional[Dict[str, Any]],
                       description="Additional method arguments")
            return config


-------------------------------
``MetricPipeline``
-------------------------------

In many cases, we may require more than a single ``Metric``.

Cinnamon adopts the nesting paradigm for metrics as well since a ``Metric`` is a ``Component``.

A simple way to wrap multiple metrics into a single parent ``Metric`` and execute them sequentially is the ``MetricPipeline`` (see :ref:`pipeline` for more details about pipelines).

.. code-block:: python

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

The ``run`` method executes all child ``Metric`` components in sequential fashion with the specified input.