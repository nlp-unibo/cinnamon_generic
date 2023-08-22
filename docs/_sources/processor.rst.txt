.. _processor:

Processor
*************************************

A ``Processor`` is a ``Component`` that is specialized in processing input data.

It is mainly used to prepare input data for a model after having loaded it via a ``DataLoader`` component.
Nonetheless, a ``Processor`` can be used to process any kind of data.

.. code-block:: python

    data = ...
    data = processor.run(data=data)

The ``Processor`` provides the following APIs:

- ``process``: processes input data.
- ``finalize``: finalizes the ``Processor`` internal state. This function should be called when no other run() function calls are required.


----------------------------------
``ProcessorPipeline``
----------------------------------

In many cases, we may have to invoke multiple processor for given input data.

Cinnamon adopts the nesting paradigm for processors as well since a ``Processor`` is a ``Component``.

A simple way to wrap multiple processors into a single parent ``Processor`` and execute them sequentially is the ``ProcessorPipeline`` (see :ref:`pipeline` for more details about pipelines).

.. code-block:: python

    class ProcessorPipeline(OrderedPipeline, Processor):

        def finalize(
                self
        ):
            components = self.get_pipeline()
            for component in components:
                component.finalize()

        def run(
                self,
                data: Optional[FieldDict] = None,
                is_training_data: bool = False
        ) -> FieldDict:
            components = self.get_pipeline()
            for component in components:
                data = component.run(data=data, is_training_data=is_training_data)
            return data

The ``run`` method executes all child ``Processor`` components in sequential fashion with the specified input data.


*************************************
Routine Processors
*************************************

A ``RoutineProcessor`` is a ``Processor`` specialized for handling ``Routine`` output results.

Cinnamon provides the following implementations:

- ``AverageProcessor``: computes average and std for each loss and metric.
- ``FoldProcessor``: computes average and std for each loss and metric over cross-validation folds.