.. _data_splitter:

Data Splitter
*************************************

A ``DataSplitter`` component splits input data into multiple splits.

Usually, these splits are `train`, `validation`, and `test`.

In cinnamon, we distinguish between the ``DataSplitter`` component and internal data splitting method.

Generally speaking, the code logic behind data splitting are a few, while there might exist several internal data splitting methods.

Cinnamon offers the following ``DataSplitter`` components

- ``TTSplitter``: train and test data splitter component.
- ``CVSplitter``: an extension of ``TTSplitter`` specialized for cross-validation fold splits.
- ``PrebuiltCVSplitter``: an extension of ``CVSplitter`` that supports saving/loading pre-built fold splits.

and the following internal data splitting methods

- ``SklearnTTSplitter``: a sklearn-compliant internal data splitter.


*************************************
Supported ``DataSplitter``
*************************************

-------------------------------------
``TTSplitter``
-------------------------------------

Splits into train, validation and test splits based on input available splits.

The splitter operates as follows:

- Training data split is required to perform any kind of splitting.
- If both validation and test splits are not specified, these splits are built based on train data.
- if validation or test splits is missing, the split is built based on train data.

The ``TTSplitter`` leverages an internal splitter method (e.g., ``SklearnTTSplitter``) to perform the splitting.

The ``TTSplitter`` uses the ``TTSplitterConfig`` as default configuration template:

.. code-block:: python

    class TTSplitterConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='validation_size',
                       type_hint=Optional[float],
                       description='Training set percentage to use as validation split')

            config.add(name='test_size',
                       type_hint=Optional[float],
                       description='Training set percentage to use as test split')

            config.add(name='splitter_type',
                       type_hint=InternalTTSplitter,
                       description='Splitter class for performing data split',
                       is_required=True)

            config.add(name='splitter_args',
                       value={},
                       description="Arguments for creating a splitter instance")

            return config

In particular, a ``SklearnTTSplitterConfig`` that uses the ``SklearnTTSplitter`` is already provided:

.. code-block:: python

    class SklearnTTSplitterConfig(TTSplitterConfig):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()
            config.splitter_type = SklearnTTSplitter
            config.splitter_args = {
                'random_state': 42,
                'shuffle': True
            }
            return config


-------------------------------------
``CVSplitter``
-------------------------------------

Builds train, validation and test fold splits based on input available splits.

The splitter operates as follows:

- Training data split is required to perform any kind of splitting.
- If both validation and test splits are not specified, these splits are built based on train data.
- if validation or test splits is missing, the split is built based on train data. The specified data split is fixed for all folds.

The ``CVSplitter`` uses the ``CVSplitterConfig`` as default configuration template:

.. code-block:: python

    class CVSplitterConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='splitter_type',
                       value=KFold,
                       type_hint=_BaseKFold,
                       description='Splitter class for performing data split',
                       is_required=True)

            config.add(name='splitter_args',
                       value={
                           'n_splits': 5,
                           'shuffle': True

                       },
                       description="Arguments for creating a splitter instance")

            config.add(name='X_key',
                       type_hint=Hashable,
                       description='Column name for input data')

            config.add(name='y_key',
                       type_hint=Any,
                       description="Column name for output data",
                       is_required=True)

            config.add(name='group_key',
                       type_hint=Hashable,
                       description='Column name for grouping')

            config.add(name='held_out_key',
                       value='validation',
                       allowed_range=lambda value: value in ['validation', 'test'],
                       type_hint=str,
                       description="Which data split key (e.g., test, validation) built folds belong to")

            config.add(name='validation_n_splits',
                       value=config.splitter_args['n_splits'],
                       type_hint=int,
                       description="Number of splits to perform to build folds",
                       is_required=True)

            return config


-------------------------------------
``PrebuiltCVSplitter``
-------------------------------------

Extends ``CVSplitter`` to allow saving/loading pre-built fold splits.

The ``PrebuiltCVSplitter`` first looks at the specified folds path. If the file exists, the pre-built folds are loaded and iterated through.
Otherwise, it operates like ``CVSplitter``.

The ``PrebuiltCVSplitter`` uses the ``PrebuiltCVSplitterConfig`` as default configuration template:

.. code-block:: python

    class PrebuiltCVSplitterConfig(CVSplitterConfig):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='prebuilt_filename',
                       type_hint=str,
                       description="Filename for storing/loading pre-built folds",
                       is_required=True)

            config.add(name='prebuilt_folder_name',
                       value='prebuilt_folds',
                       type_hint=Union[AnyStr, Path],
                       description="Folder name where to store pre-built fold files",
                       is_required=True)

            config.add(name='file_manager_key',
                       value=RegistrationKey(name='file_manager',
                                             tags={'default'},
                                             namespace='generic'),
                       description="Registration key pointing to built FileManager component",
                       is_required=True)

            return config

*************************************
Supported internal data splitters
*************************************

-------------------------------------
``SklearnTTSplitter``
-------------------------------------

The ``SklearnTTSplitter`` simply wraps the ``train_test_split`` function of `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_.

.. note::
    The ``SklearnTTSplitter is not a ``Component``! It is just an ordinary python class.