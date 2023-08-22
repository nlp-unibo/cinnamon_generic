.. _pipeline:

Pipeline
*************************************

In cinnamon, ``Component`` and ``Configuration`` can be nested to define more advanced code logics and related configurations.

In some cases, we may only need to wrap many components altogether and executes them in sequence, possibly in a specified order.

To do so, cinnamon defines the ``Pipeline`` component.

Currently, cinnamon supports the following implementations:

- ``Pipeline``: executes components in a sequential fashion without a particular order.
- ``OrderedPipeline``: executes components in a specified sequential order.

We can quickly wrap multiple components in a ``Pipeline`` via ``Pipeline.from_components()`` method:

..  code-block:: python

    pipeline = Pipeline.from_components([c1, c2, ..., cN])
    pipeline.run()                                              # runs all pipeline components in a sequential fashion
    components = pipeline.get_pipeline()                        # returns [c1, c2, ..., cN]



---------------------------
``Pipeline``
---------------------------

The ``Pipeline`` uses the ``PipelineConfig`` as the default configuration template.

In particular, the ``PipelineConfig`` allows to quickly wrap multiple configurations in a ``PipelineConfig`` via ``PipelineConfig.from_keys()`` method:

.. code-block:: python

    pipeline_config = PipelineConfig.from_keys([RegistrationKey(...), ..., RegistrationKey(...)])


The ``PipelineConfig`` also allows to quickly add new pipeline configurations via ``add_pipeline_component`` just like any other ``Parameter``.

.. code-block:: python

    config.add_pipeline_component(name='my_component',
                                  value=RegistrationKey(...),
                                  description=...,
                                  tags=...,
                                  variants=...)


---------------------------
``OrderedPipeline``
---------------------------

The ``OrderedPipeline`` is an extension of ``Pipeline`` that supports execution order.

The ``OrderedPipeline`` uses the ``OrderedPipelineConfig`` as the default configuration template.

In particular, ``OrderedPipelineConfig`` extends ``add_pipeline_component`` to also include a ``order`` field.