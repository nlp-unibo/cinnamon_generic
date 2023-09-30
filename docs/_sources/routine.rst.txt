.. _routine:

Routine
*************************************

A ``Routine`` is a ``Component`` specialized in defining an evaluation routine.

The ``Routine`` can be viewed as the highest-level ``Component`` since it wraps multiple components: from data loading to modeling.

A ``Routine`` may involve repeating the same execution flow with different input data.
For instance:

- **Train and test**: repeats train and test setting for different random seeds
- **Cross-validation**: repeat train and test setting for different inputs and (potentially) different random seeds

.. note::
    Other evaluation criteria like leave-one-out are included in cross-validation as they are special cases.

The ``Routine`` uses ``RoutineConfig`` as the default configuration template.

.. code-block:: python

    class RoutineConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config: Configuration = super().get_default()

            config.add(name='seeds',
                       type_hint=Union[List[int], int],
                       description="Seeds to use for reproducible benchmarks",
                       is_required=True)

            config.add(name='data_loader',
                       type_hint=RegistrationKey,
                       build_type_hint=DataLoader,
                       description="DataLoader component used to for data loading",
                       is_required=True,
                       is_child=True)

            config.add(name='data_splitter',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=TTSplitter,
                       description="Data splitter component for creating train/val/test splits",
                       is_required=True,
                       is_child=True)

            config.add(name='pre_processor',
                       type_hint=RegistrationKey,
                       build_type_hint=Processor,
                       description="Processor component used to for data pre-processing",
                       is_required=True,
                       is_child=True)

            config.add(name='post_processor',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Optional[Processor],
                       description="Processor component used to for data post-processing",
                       is_child=True)

            config.add(name='model_processor',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Optional[Processor],
                       description="Processor component used to for model output post-processing",
                       is_child=True)

            config.add(name='routine_processor',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Optional[Processor],
                       description="Processor component used to for routine results processing",
                       is_child=True)

            config.add(name='model',
                       type_hint=RegistrationKey,
                       build_type_hint=Model,
                       description="Model component used to wrap a machine learning model ",
                       is_required=True,
                       is_child=True)

            config.add(name='callbacks',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Callback,
                       description="Callback component for customized control flow and side effects",
                       is_child=True)

            config.add(name='metrics',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Metric,
                       description="Metric component for routine evaluation",
                       is_child=True)

            config.add(name='helper',
                       type_hint=Optional[RegistrationKey],
                       build_type_hint=Optional[Helper],
                       description="Helper component for reproducibility and backend management",
                       is_child=True)

            return config

.. note::
    The ``Routine`` is an example of nested ``Component``!


Cinnamon provides the following ``Routine`` implementations:

- ``TrainAndTestRoutine``: implements the 'Train and Test' evaluation routine.
- ``CVRoutine``: implements the 'cross-validation' evaluation routine.

We can run a ``Routine`` via its ``run()`` method.

.. code-block:: python

    routine.run(is_training=True)       # Training mode
    routine.run(is_training=False)      # Inference mode