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


*************************************
NLP Text Processors
*************************************

The ``cinnamon-generic`` package also provides NLP-specific ``Processor``.

Cinnamon provides the following implementations:

- ``TextProcessor``: A ``Processor`` for processing general-purpose text.
- ``TokenizerProcessor``: A ``Processor`` specialized in processing text data via a tokenizer.


-----------------------------
``TextProcessor``
-----------------------------

The ``TextProcessor`` is a general-purpose text processor that can support multiple input texts.

Input texts are retrieved from input ``FieldDict`` by search ``Field`` with ``text`` tag.
Subsequently, each text input is processed and the updated ``FieldDict`` is returned.

.. warning::
    The ``TextProcessor`` performs updates input data text fields in place.

The ``TextProcessor`` uses ``TextProcessorConfig`` as default configuration template.

.. code-block:: python

    class TextProcessorConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='filters',
                       affects_serialization=True,
                       type_hint=Optional[List[Callable]],
                       description='List of filter functions that accept a text as input')

            return config

Filters are applied in sequence to each text sample of each text field in input data.

-----------------------------
``TokenizerProcessor``
-----------------------------

The ``TokenizerProcessor`` processes input data via an internal tokenizer.
If ``is_training_data = True``, input data is used to first fit the tokenizer.

Similarly to ``TextProcessor``, ``TokenizerProcessor`` retrieves all text fields via ``text`` tag and feeds them to its internal tokenizer.

Moreover, ``TokenizerProcessor`` supports pre-trained embedding models loading to build the corresponding embedding matrix.
In particular, the embedding matrix is computed after ``finalize()`` is invoked.

.. note::
    This is done since the ``TokenizerProcessor`` may fit on multiple data sources. It is the user's responsibility to notify ``TokenizerProcessor`` when the embedding matrix can be computed by invoking ``finalize()``.

The ``TokenizerProcessor`` uses ``TokenizerProcessorConfig`` as default configuration template.

.. code-block:: python

    class TokenizerProcessorConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='fit_on_train_only',
                       value=True,
                       affects_serialization=True,
                       type_hint=bool,
                       description='If disabled, the tokenizer builds its vocabulary on all available data')

            config.add(name='merge_vocabularies',
                       value=False,
                       affects_serialization=True,
                       type_hint=bool,
                       description="If enabled, the pre-trained embedding model and input "
                                   "data vocabularies are merged.")

            config.add(name='embedding_dimension',
                       value=50,
                       affects_serialization=True,
                       type_hint=int,
                       description='Embedding dimension for text conversion')

            config.add(name='embedding_type',
                       affects_serialization=True,
                       type_hint=Optional[str],
                       description='Pre-trained embedding model type (if any)',
                       allowed_range=lambda emb_type: emb_type in {
                           "word2vec-google-news-300",
                           "glove-wiki-gigaword-50",
                           "glove-wiki-gigaword-100",
                           "glove-wiki-gigaword-200",
                           "glove-wiki-gigaword-300",
                           "fasttext-wiki-news-subwords-300"
                       })

            config.add_condition(name='valid_embedding_model',
                                 condition=lambda parameters: parameters.embedding_type is None
                                                              or (parameters.embedding_type is not None
                                                                  and str(parameters.embedding_dimension)
                                                                  in parameters.embedding_type))

            return config


***************************
Registered configurations
***************************

The ``cinnamon-generic`` package provides the following registered configurations:

- ``name='processor', tags={'text'}, namespace='generic'``: the default ``TextProcessor``.
- ``name='processor', tags={'text', 'tokenizer'}, namespace='generic'``: the default ``TokenizerProcessor``.
