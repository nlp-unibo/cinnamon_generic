from typing import List, Callable, Optional

from cinnamon_core.core.registry import Registry

from cinnamon_generic.configurations.calibrator import TunableConfiguration
from cinnamon_generic.nlp.components.processor import TextProcessor, TokenizerProcessor


class TextProcessorConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.add_short(name='filters',
                         affects_serialization=True,
                         type_hint=Optional[List[Callable]],
                         description='List of filter functions that accept a text as input')

        return config


class TokenizerProcessorConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='fit_on_train_only',
                         value=True,
                         affects_serialization=True,
                         type_hint=bool,
                         description='If disabled, the tokenizer builds its vocabulary on all available data')
        config.add_short(name='merge_vocabularies',
                         value=False,
                         affects_serialization=True,
                         type_hint=bool,
                         description="If enabled, the pre-trained embedding model and input "
                                     "data vocabularies are merged.")
        config.add_short(name='embedding_dimension',
                         value=50,
                         affects_serialization=True,
                         type_hint=int,
                         description='Embedding dimension for text conversion')
        config.add_short(name='embedding_type',
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


def register_processors():
    Registry.register_and_bind(configuration_class=TextProcessorConfig,
                               component_class=TextProcessor,
                               name='processor',
                               tags={'text'},
                               is_default=True)
    Registry.register_and_bind(configuration_class=TokenizerProcessorConfig,
                               component_class=TokenizerProcessor,
                               name='processor',
                               tags={'text', 'tokenizer'},
                               is_default=True)
