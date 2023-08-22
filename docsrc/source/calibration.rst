.. _calibration:

Calibrator
*************************************

The ``Calibrator`` component perform hyper-parameter calibration of a given ``Component``, known as **validator**.

The ``Calibrator`` retrieves the hyper-parameter search space of the **validator** and runs its hyper-parameter combinations to find the best one according to the specified validation metric.

The following ``Calibrator`` implementations are provided:

- ``GridSearchCalibrator``: performs grid-search hyper-parameter calibration.
- ``RandomSearchCalibrator``: performs random hyper-parameter calibration.
- ``HyperOptCalibrator``: uses `hyperopt <https://hyperopt.github.io/hyperopt/>`_ for hyper-parameter calibration.

The general ``CalibratorConfig`` has the following default template

.. code-block:: python

    class CalibratorConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
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

-----------------------------------------
Defining a hyper-parameter search space
-----------------------------------------

Cinnamon has a simple method to allow plug-and-play search spaces for a particular ``Component``: a ``Configuration`` defines the search space and is set as a child ``Configuration`` of the component's ``Configuration``.

In particular, ``cinnamon-generic`` defines a ``TunableConfiguration``, a ``Configuration`` with a ``calibration_config`` parameter that points to the search space ``Configuration``.

In other terms, we first define a ``Configuration`` to specify the search space as follows

.. code-block:: python

    class CalibrationConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='search_space',
                       value={
                           'param1': [1, 2, 3],
                           'param2': [False, True]
                       })
            return config

.. note::
    The ``search_space`` parameter is the **one** and **only** parameter needed.

Subsequently, we define our component's ``Configuration``, inheriting from ``TunableConfiguration``:

.. code-block:: python

    class MyConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='param1',
                       value=1,
                       type_hint=int)
            config.add(name='param2',
                       value=True,
                       type_hint=bool)
            config.calibration_config = RegistrationKey(name='calibration',
                                                        tags={'config_a'},
                                                        namespace='testing')

            return config


 .. note::
    We have to set the ``calibration_config`` parameter  to point to our **registered** calibration configuration.

We can quickly retrieve the search space of ``MyConfig`` by invoking the ``get_search_space()`` instance method

.. code-block:: python

    config = MyConfig.get_default()
    print(config.get_search_space())
    # {'param1': [1, 2, 3], 'param2': [False, True]}

De-coupling the search space of a ``Configuration`` to its template allows defining multiple search space configurations independently of their implementation!
For instance, we can quickly change ``calibration_config`` to another ``RegistrationKey`` that defines the search space via ``hyperopt`` or other ad-hoc packages.

The ``get_search_space()`` method **supports nesting**! Thus, we can quickly retrieve the complete search space of our nested ``Component``.

.. code-block:: python

   class ConfigA(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='param1',
                       value=1,
                       type_hint=int)
            config.add(name='param2',
                       value=True,
                       type_hint=bool)
            config.add(name='child',
                       value=RegistrationKey(name='config_b',
                                             namespace='testing'),
                       is_registration=True)
            config.calibration_config = RegistrationKey(name='calibration',
                                                        tags={'config_a'},
                                                        namespace='testing')

            return config

    class CalibrationConfigA(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='search_space',
                       value={
                           'param1': [1, 2, 3],
                           'param2': [False, True]
                       })
            return config


    class ConfigB(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='param1',
                       value=True,
                       type_hint=bool)
            config.calibration_config = RegistrationKey(name='calibration',
                                                        tags={'config_b'},
                                                        namespace='testing')
            return config

    class CalibrationConfigB(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.add(name='search_space',
                       value={
                           'param1': [False, True]
                       })
            return config

If we now call ``get_search_space()``, we get

.. code-block:: python

    config = ConfigA.get_default()
    print(config.get_search_space())
    # {'param1': [1, 2, 3], 'param2': [False, True], 'child.param1': [False, True]}

.. note::
    Note how ``param1`` of ``ConfigB`` is referenced by its parent ``ConfigA``. This how cinnamon keeps track of nested parameters.
    The ``Calibrator`` supports dictionaries in this form to build **validator** variants.

--------------------------------------
GridSearchCalibrator
--------------------------------------

Performs grid-search hyper-parameter calibration.

The grid-search samples and evaluates all possible hyper-parameter combinations of the **validator**.

The ``GridSearchCalibrator`` uses the ``CalibratorConfig`` as default configuration.


--------------------------------------
RandomSearchCalibrator
--------------------------------------

Performs random search hyper-parameter calibration.

The ``RandomSearchCalibrator`` samples validator combinations until the specified maximum number of ``tries`` is reached.

The ``RandomSearchCalibrator`` uses the ``RandomSearchCalibratorConfig`` as default configuration.

.. code-block:: python

    class RandomSearchCalibratorConfig(CalibratorConfig):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='tries',
                       value=10,
                       type_hint=int,
                       allowed_range=lambda value: value >= 1,
                       is_required=True,
                       description='Number of hyper-parameter combinations to randomly sample and try')

            return config


--------------------------------------
HyperOptCalibrator
--------------------------------------

Wraps a ``hyperopt`` calibrator to perform hyper-parameter calibration of any ``Component``.

The current implementation of the ``HyperOptCalibrator`` supports **sequential** and **mongodb** implementations (see the `hyperopt <https://hyperopt.github.io/hyperopt/>`_ official documentation page for more details)

In particular, the **mongodb** version requires a running mongodb server. Mongo workers are automatically executed and handled by the component, instead.

The ``HyperOptCalibrator`` uses the ``HyperOptCalibratorConfig`` as default configuration.

.. code-block:: python

    class HyperoptCalibratorConfig(CalibratorConfig):

        @classmethod
        def get_default(
                cls
        ):
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

            # Conditions
            config.add_condition(name='worker_sleep_interval_minimum',
                                 condition=lambda parameters: parameters.worker_sleep_interval.value >= 0.5)

            config.add_condition(name="max_evaluations_minimum",
                                 condition=lambda parameters: parameters.max_evaluations > 0)

            return config