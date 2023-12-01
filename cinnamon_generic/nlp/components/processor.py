import abc
from functools import reduce
from typing import Union, List, Dict, Iterable, Optional

import gensim
import gensim.downloader as gloader
import numpy as np
from tqdm import tqdm

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.processor import Processor


class TextProcessor(Processor):
    """
    A ``Processor`` specialized in processing general-purpose text.
    """

    def apply_chain_filters(
            self,
            text: str
    ) -> str:
        """
        Applies a chain of text filters to input text.

        Args:
            text: input text to be processed

        Returns:
            Processed text
        """

        return reduce(lambda r, f: f(r), self.filters, text)

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        """
        Processes text inputs from input data. Texts inputs are retrieved by ``text`` tag from
         input ``data`` ``FieldDict``.

        Args:
            data: input data containing text fields to be processed
            is_training_data: if True, input data is from a training split

        Returns:
            Input data with processed text fields.
        """

        text_fields = data.get_by_tag(tags='text')
        for field in text_fields:
            data[field.name] = [self.apply_chain_filters(text=item) for item in field.value]
        return data


class TokenizerProcessor(Processor):
    """
    A ``Processor`` specialized in processing text data via a tokenizer.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_model: Optional[gensim.models.keyedvectors.KeyedVectors] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        self.vocabulary: Optional[Dict] = None
        self.vocab_size: Optional[int] = None

    def prepare_save_data(
            self
    ) -> Dict:
        data = super().prepare_save_data()

        data['embedding_model'] = self.embedding_model
        data['embedding_matrix'] = self.embedding_matrix
        data['vocabulary'] = self.vocabulary
        data['vocab_size'] = self.vocab_size

        return data

    @abc.abstractmethod
    def fit(
            self,
            data: FieldDict,
    ):
        """
        Fits the internal tokenizer based on input textual data

        Args:
            data: input data containing text fields to be parsed by the tokenizer
        """
        pass

    def finalize(
            self
    ):
        """
        If specified, it loads embedding model and computes pre-trained embedding matrix.
        """

        if self.embedding_type is not None:
            self.load_embedding_model()

            if self.embedding_model is None:
                raise RuntimeError(f'Expected a pre-trained embedding model. Got {self.embedding_model}')

            self.build_embeddings_matrix(vocabulary=self.vocabulary,
                                         embedding_model=self.embedding_model,
                                         embedding_dimension=self.embedding_dimension)

    def clear(
            self
    ):
        super().clear()
        self.embedding_model = None
        self.embedding_matrix = None
        self.vocabulary = None
        self.vocab_size = None

    def load_embedding_model(
            self,
    ):
        """
        Loads a pre-trained word embedding model via Gensim library.
        """
        logging_utility.logger.info(f'Loading pre-trained embedding model: {self.embedding_type}')
        embedding_model = gloader.load(self.embedding_type)
        self.embedding_model = embedding_model

    def build_embeddings_matrix(
            self,
            vocabulary: Dict[str, int],
            embedding_model: gensim.models.keyedvectors.KeyedVectors,
            embedding_dimension: int = 300,
    ):
        """
        Builds embedding matrix given the pre-trained embedding model.

        Args:
            vocabulary: the tokenizer vocabulary after fitting
            embedding_model: the pre-trained embedding model loaded via Gensim model.
            embedding_dimension: the embedding dimension

        Returns:
            The built embedding matrix and the updated vocabulary
        """
        added_tokens = []

        if self.merge_vocabularies:
            vocab_size = len(set(list(vocabulary.keys()) + list(embedding_model.vocab.keys()))) + 1
            for key in tqdm(embedding_model.vocab.keys()):
                if key not in vocabulary:
                    vocabulary[key] = max(list(vocabulary.values())) + 1
                    added_tokens.append(key)
        else:
            vocab_size = self.vocab_size

        embedding_matrix = np.zeros((vocab_size, embedding_dimension))
        for word, i in tqdm(vocabulary.items()):
            try:
                embedding_vector = embedding_model[word]
                # Check for any possible invalid term
                if embedding_vector.shape[0] != embedding_dimension:
                    embedding_vector = np.zeros(embedding_dimension)
            except (KeyError, TypeError):
                embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

            embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix
        return vocabulary, added_tokens

    @abc.abstractmethod
    def tokenize(
            self,
            text: Iterable[str],
            remove_special_tokens: bool = False
    ) -> Union[List[int], np.ndarray]:
        """
        Tokenizes input text via the fitted tokenizer.

        Args:
            text: input text to be tokenized
            remove_special_tokens: if True, the tokenizer will not report any special tokens.

        Returns:
            The tokenized input text
        """
        pass

    @abc.abstractmethod
    def detokenize(
            self,
            ids: Iterable[Union[List[int], np.ndarray]],
            remove_special_tokens: bool = False
    ) -> str:
        """
        Converts input tokens back to text format via the fitted tokenizer.

        Args:
            ids: input token ids to be converted back to text format
            remove_special_tokens: if True, the tokenizer will not report any special tokens

        Returns:
            The detokenized text from input token ids.
        """
        pass

    def process(
            self,
            data: FieldDict,
            tokenize: bool = True,
            remove_special_tokens: bool = False):
        """
        Processes input data via an internal tokenizer.
        Each text field in input data is parsed by the tokenizer.
        The ``tokenize`` argument dictates whether the tokenizer should tokenize or de-tokenize input data.

        Args:
            data: input data containing text fields to be parsed by the tokenizer
            tokenize: if True, the tokenizer will tokenize text fields. Otherwise, de-tokenization will be carried out.
            remove_special_tokens: if True, the tokenizer will not report any special tokens

        Returns:
            The input data with text fields processed by the internal tokenizer.
        """

        text_fields = data.search_by_tag(tags='text')
        for field_name, field_value in text_fields.items():
            if tokenize:
                value = self.tokenize(text=field_value,
                                      remove_special_tokens=remove_special_tokens)
            else:
                value = self.detokenize(ids=field_value,
                                        remove_special_tokens=remove_special_tokens)
            data[field_name] = value
        return data

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False,
            tokenize: bool = True,
            remove_special_tokens: bool = False,
            **kwargs
    ) -> Optional[FieldDict]:
        """
        Processes input data via an internal tokenizer.
        If ``is_training_data = True``, input data is used to first fit the tokenizer.

        Args:
            data: input data containing text fields to be parsed by the tokenizer
            is_training_data: if True, input data is from a training split
            tokenize: if True, the tokenizer will tokenize text fields. Otherwise, de-tokenization will be carried out.
            remove_special_tokens: if True, the tokenizer will not report any special tokens

        Returns:
            The input data with text fields processed by the internal tokenizer.ut
        """
        if is_training_data or not self.fit_on_train_only:
            if data is None:
                if is_training_data:
                    raise AttributeError(f'Expected to fit on some training data. Got {data}')
                else:
                    return data

            self.fit(data=data)

            if self.vocabulary is None:
                raise RuntimeError(f'Expected vocabulary to be not None. Got {self.vocabulary}')

        if data is not None:
            data = self.process(data=data,
                                tokenize=tokenize,
                                remove_special_tokens=remove_special_tokens)

        return data
