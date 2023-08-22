.. _commands:

Commands
*************************************

Commands are high-level APIs that wrap some boilerplate code in cinnamon, such as registry initialization and component running.

Currently, the following commands are provided:

- ``setup_registry``
- ``run_component_from_key`` and ``run_component``
- ``run_components``
- ``routine_train`` and ``routine_train_from_key``
- ``routine_multiple_train``
- ``routine_inference``
- ``routine_multiple_inference``
- ``run_calibration`` and ``run_calibration_from_key``

-------------------------------------
Initializing the registry
-------------------------------------

This command does the following actions:

- Populates the ``Registry`` with specified registration actions.
- Builds the ``FileManager`` ``Component`` and stores its instance in the ``Registry`` for quick use.
- Set-ups the logging utility module.
- If ``generate_registration``, invokes the ``list_registrations`` command for debugging and readability purposes.

.. warning::
    !IMPORTANT!: this command is always required at beginning of each of your scripts for proper ``Registry`` initialization.

.. code-block:: python

    from pathlib import Path
    from cinnamon_generic.api.commands import setup_registry

    if __name__ == '__main__':
        """
        In this demo script, we test ``setup_registry`` command to check if all registration actions are performed correctly.
        If ``registrations_to_file=True`` you will see a ``dependencies.html`` file in ``demos/`` folder.
        You can open it in the browser to inspect all registered configurations and their dependencies.
        """
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)


-------------------------------------
Running a Component
-------------------------------------

Consider a ``DataLoader`` component.

The code to run the component without ``run_component_from_key`` and ``run_component`` commands is as follows

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry
    from cinnamon_generic.components.data_loader import DataLoader

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        loader = DataLoader.build_component(name='data_loader',
                                            tags=...,
                                            namespace='showcasing')
        data = loader.run()


And now with commands

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, run_component
    from cinnamon_core.core.registry import Registry

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        data, _ = run_component(name='data_loader',
                                  tags=...,
                                  namespace='showcasing,
                                  run_name='showcase',
                                  serialize=False)


-------------------------------------
Running multiple Component
-------------------------------------

We can quickly run our ``Routine`` component in training mode as follows.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, routine_train

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        result = routine_train(name='routine',
                               tags=...,
                               namespace='showcasing',
                               run_name='routine_train',
                               serialize=True)


.. note::
    The ``serialize`` argument makes sure that ``Routine`` results and component's internal state are serialized for quick re-use (e.g., inference mode)

Once trained, we can issue again the same ``Routine`` in inference mode, instead.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, routine_inference

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        result = routine_inference(routine_path=...,
                                   namespace='showcasing',
                                   run_name='routine_test',
                                   serialize=False)

This commands repeats the same routine execution flow, where employed models are loaded from filesystem.

.. note::
    No training is performed, just model predictions on data splits.


------------------------------------
Running multiple Routine
------------------------------------

Running more than a ``Component`` in a sequential fashion is straightforward.
We provide an example with ``Routine`` using ``routine_multiple_train`` command.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_generic.api.commands import setup_registry, routine_multiple_train

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        result = routine_multiple_train(routine_keys=[RegistrationKey(...), ..., RegistrationKey(...)],
                                        serialize=True)


-------------------------------------
Running a Calibrator
-------------------------------------

Running a ``Calibrator`` is as simple as running any other ``Component``.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, run_calibration

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        result = run_calibration(name='calibrator',
                                 tags=...,
                                 namespace='showcasing',
                                 serialize=False)


*************************************
Runners
*************************************

Running a command may be cumbersome since it requires manual code change.
Additionally, for commands like ``run_multiple_routine_train`` we may rely on ``Registry`` to retrieve a subset of ``RegistrationKey``: this operation might require custom python code.
For instance:

.. code-block:: python

     keys = [key for key in Registry.REGISTRY if key.name == 'routine' and 'tag1' in key.tags and 'tag2' in key.tags and 'tag3' not in key.tags]


For this reason, cinnamon introduces command runners: command-specific components that store signature arguments.
Command runners also support argument parsing to quickly run scripts from terminal.

Command runners are useful since they only require their ``RegistrationKey`` to be retrieved, and thus, run a particular script.
This functionality is particularly useful when running scripts from terminal.

Currently, cinnamon provides the following command runners:

- ``ComponentRunner``
- ``MultipleComponentRunner``
- ``MultipleRoutineTrainRunner``
- ``RoutineInferenceRunner``


-----------------------
``ComponentRunner``
-----------------------

Stores ``name``, ``tags``, ``namespace`` and ``run_name`` arguments.
It can be used with commands like ``run_component``.

The ``ComponentRunner`` uses ``ComponentRunnerConfig`` as the default configuration template.

.. code-block:: python

    class ComponentRunnerConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='name',
                       type_hint=str,
                       is_required=True)

            config.add(name='tags',
                       type_hint=Tag)

            config.add(name='namespace',
                       type_hint=str,
                       is_required=True)

            config.add(name='run_name',
                       type_hint=str)

            return config


For instance, the script for training a ``Routine`` changes to:

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, routine_train

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        runner = Registry.build_component(name='command',
                                          tags=...,
                                          namespace='showcasing')
        cmd_config = runner.run()

        result = routine_train(name=cmd_config.name,
                               tags=cmd_config.tags,
                               namespace=cmd_config.namespace,
                               serialize=True,
                               run_name=cmd_config.run_name)


-----------------------------
``MultipleComponentRunner``
-----------------------------

Stores ``registration_keys``, ``runs_names``, and ``run_args`` arguments.
It can be used with commands like ``run_multiple_components``.

The ``MultipleComponentRunner`` uses ``MultipleComponentRunnerConfig`` as the default configuration template.

.. code-block:: python

    class MultipleComponentRunnerConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='registration_keys',
                       type_hint=Union[List[Registration], Callable[[], List[Registration]]],
                       is_required=True)

            config.add(name='runs_names',
                       type_hint=Optional[List[str]])

            config.add(name='run_args',
                       type_hint=Optional[List[Dict]])

            return config


--------------------------------
``MultipleRoutineTrainRunner``
--------------------------------

Stores ``routine_keys``.
It can be used with``run_multiple_routine_train``.

The ``MultipleRoutineTrainRunner`` uses ``MultipleRoutineTrainRunnerConfig`` as the default configuration template.

.. code-block:: python

    class MultipleRoutineTrainRunnerConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='routine_keys',
                       type_hint=Union[List[Registration], Callable[[], List[Registration]]],
                       is_required=True)
            return config


For instance, we can register a ``MultipleRoutineTrainRunnerConfig`` on-the-fly as follows

.. code-block:: python

    Registry.add_and_bind(config_class=MultipleRoutineTrainRunnerConfig,
                              component_class=MultipleRoutineTrainRunner,
                              config_constructor=MultipleRoutineTrainRunnerConfig.get_delta_class_copy,
                              config_kwargs={
                                  'params': {
                                      'routine_keys': lambda: [key for key in Registry.REGISTRY
                                                               if key.name == 'routine' and
                                                               'tag1' in key.tags and
                                                               'tag2' in key.tags and
                                                               'tag3' not in key.tags and
                                                               Registry.build_configuration_from_key(key).validate(
                                                                   strict=False).passed
                                                               ]
                                  }
                              },
                              name='command',
                              tags=...,
                              namespace='showcasing')


And use it in our script

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_generic.api.commands import setup_registry, routine_multiple_train

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        runner = Registry.build_component(name='command',
                                          tags=...,
                                          namespace='showcasing')
        cmd_config = runner.run()
        result = routine_multiple_train(routine_keys=cmd_config.routine_keys,
                                        serialize=True)


Note how all the custom code for retrieving registration keys is delegated to the command runner.
The user now has only to remember the ``RegistrationKey`` associated with the registered command runner.

.. note::
    We may consider a command runner as a script wrapper, where the script acts as the topmost level ``Component``.

--------------------------------
``RoutineInferenceRunner``
--------------------------------

Stores ``namespace``, ``routine_path``, and ``run_name``.
It can be used with``routine_inference``.

The ``RoutineInferenceRunner`` uses ``RoutineInferenceRunnerConfig`` as the default configuration template.

.. code-block:: python

    class RoutineInferenceRunnerConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='namespace',
                       type_hint=str)

            config.add(name='routine_path',
                       type_hint=Optional[Path])

            config.add(name='run_name',
                       type_hint=str)

            return config