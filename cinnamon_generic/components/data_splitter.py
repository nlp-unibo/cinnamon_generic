import abc
from pathlib import Path
from typing import Tuple, Any, Optional, List

import pandas as pd
from sklearn.model_selection import train_test_split

from cinnamon_core.core.component import Component
from cinnamon_core.utility.pickle_utility import save_pickle, load_pickle
from cinnamon_generic.components.file_manager import FileManager


class InternalTTSplitter:
    """
    Base class for defining an internal data splitter used by a splitter ``Component``.
    """

    @abc.abstractmethod
    def split(
            self,
            data: Any,
            size: Any
    ) -> Tuple[Any, Any]:
        """
        Splits input data into two splits based on specified size.

        Args:
            data: data to split
            size: split size to define

        Returns:
            The two data splits
        """

        pass


class SklearnTTSplitter(InternalTTSplitter):
    """
    A sklearn-compliant internal splitter wrapper.
    """

    def __init__(
            self,
            **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def split(
            self,
            data: Any,
            size: Any
    ) -> Tuple[Any, Any]:
        """
        Splits input data into two splits based on specified size.

        Args:
            data: data to split
            size: split size to define

        Returns:
            The two data splits
        """

        return train_test_split(data,
                                test_size=size,
                                random_state=getattr(self, 'random_state') if hasattr(self, 'random_state') else None,
                                shuffle=getattr(self, 'shuffle') if hasattr(self, 'shuffle') else None,
                                stratify=getattr(self, 'stratify') if hasattr(self, 'stratify') else None)


class TTSplitter(Component):
    """
    Train and test data splitter component.
    The splitter leverages an internal splitter to build data splits.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.splitter = self.splitter_type(**self.splitter_args)

    def run(
            self,
            train_data: pd.DataFrame,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Splits into train, validation and test splits based on input available splits.
        The splitter operates as follows:
            - Training data split is required to perform any kind of splitting
            - If both validation and test splits are not specified, these splits are built based on train data
            - if validation or test splits is missing, the split is built based on train data

        Args:
            train_data: training data split
            val_data: optional validation data split
            test_data: optional test data split

        Returns:
            Train, validation and test data splits
        """

        if val_data is None and test_data is None:
            train_data, val_data = self.splitter.split(train_data,
                                                       size=self.validation_size)
            train_data, test_data = self.splitter.split(train_data,
                                                        size=self.test_size)
            return train_data, val_data, test_data

        if val_data is None:
            train_data, val_data = self.splitter.split(train_data,
                                                       size=self.validation_size)
            return train_data, val_data, test_data

        train_data, test_data = self.splitter.split(train_data,
                                                    size=self.test_size)
        return train_data, val_data, test_data


class CVSplitter(TTSplitter):
    """
    An extension of ``TTSplitter`` to define cross-validation fold splits.
    """

    def get_split_input(
            self,
            data: pd.DataFrame
    ) -> Tuple[Optional[Any], Any, Optional[Any]]:
        X = data[self.X_key].values if self.X_key is not None else None
        y = data[self.y_key].values
        groups = data[self.group_key].values if self.group_key is not None else None

        return X, y, groups

    def held_out_split(
            self,
            X: Any,
            y: Any,
            groups: Optional[Any] = None,
    ):
        for train_indexes, held_out_indexes in self.splitter.split(X, y, groups):
            if self.held_out_key == 'validation':
                yield train_indexes, held_out_indexes, None
            else:
                yield train_indexes, None, held_out_indexes

    def all_split(
            self,
            X: Any,
            y: Any,
            groups: Optional[Any] = None,
    ):
        """
        Builds train, validation and test fold splits.

        Returns:
            A generator of train, validation and test fold splits
        """

        for train_indexes, held_out_indexes in self.splitter.split(X, y, groups):
            sub_X = X[train_indexes]
            sub_y = y[train_indexes]
            sub_groups = groups[train_indexes] if groups is not None else None

            n_splits = self.splitter.n_splits
            self.splitter.n_splits = self.validation_n_splits
            sub_train_indexes, sub_val_indexes = list(self.splitter.split(sub_X, sub_y, sub_groups))[0]
            self.splitter.n_splits = n_splits

            yield train_indexes[sub_train_indexes], train_indexes[sub_val_indexes], held_out_indexes

    def run(
            self,
            train_data: pd.DataFrame,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Builds train, validation and test fold splits based on input available splits.
        The splitter operates as follows:
            - Training data split is required to perform any kind of splitting
            - If both validation and test splits are not specified, these splits are built based on train data
            - if validation or test splits is missing, the split is built based on train data. The specified
            data split is fixed for all folds.

        Args:
            train_data: training data
            val_data: optional validation data
            test_data: optional test data

        Returns:
            Train, validation and test data fold splits
        """

        X, y, groups = self.get_split_input(data=train_data)

        if val_data is None and test_data is None:
            splits = self.all_split(X=X, y=y, groups=groups)

            for train_indexes, val_indexes, test_indexes in splits:
                yield train_data.iloc[train_indexes], train_data.iloc[val_indexes], train_data.iloc[test_indexes]
        else:
            if val_data is not None and self.held_out_key == 'validation':
                raise RuntimeError(f'Cannot define validation split if val_data is given.')
            splits = self.held_out_split(X=X, y=y, groups=groups)

            for train_indexes, val_indexes, test_indexes in splits:
                if self.held_out_key == 'validation':
                    yield train_data.iloc[train_indexes], train_data.iloc[val_indexes], test_data
                else:
                    yield train_data.iloc[train_indexes], val_data, train_data.iloc[test_indexes]


class PrebuiltCVSplitter(CVSplitter):
    """
    An extension of ``CVSplitter`` that supports saving/loading pre-built fold splits.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.folds_path: Optional[Path] = None
        self.folds: List = []

    def held_out_split(
            self,
            X: Any,
            y: Any,
            groups: Optional[Any] = None,
    ):
        if self.folds_path.exists():
            for train_indexes, val_indexes, test_indexes in self.load_folds():
                yield train_indexes, val_indexes, test_indexes
        else:
            for train_indexes, val_indexes, test_indexes in super().held_out_split(X=X, y=y, groups=groups):
                self.folds.append((train_indexes, val_indexes, test_indexes))
                yield train_indexes, val_indexes, test_indexes
            self.save_folds()

    def all_split(
            self,
            X: Any,
            y: Any,
            groups: Optional[Any] = None,
    ):
        """
        Builds train, validation and test fold splits.
        If ``folds_path`` points to an existing file, the fold splits are loaded rather than being generated.

        Returns:
            A generator of train, validation and test fold splits
        """

        if self.folds_path.exists():
            for train_indexes, val_indexes, test_indexes in self.load_folds():
                yield train_indexes, val_indexes, test_indexes
        else:
            for train_indexes, val_indexes, test_indexes in super().all_split(X=X, y=y, groups=groups):
                self.folds.append((train_indexes, val_indexes, test_indexes))
                yield train_indexes, val_indexes, test_indexes
            self.save_folds()

    def save_folds(
            self,
    ):
        """
        Saves built cross-validation folds to filesystem for quick re-use in JSON format.
        """
        save_pickle(self.folds_path, self.folds)

    def load_folds(
            self
    ):
        """
        Loads pre-built cross-validation folds from filesystem.
        """
        self.folds = load_pickle(self.folds_path)
        return self.folds

    def run(
            self,
            train_data: pd.DataFrame,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Builds train, validation and test fold splits based on input available splits.
        The splitter operates as follows:
            - Training data split is required to perform any kind of splitting
            - If both validation and test splits are not specified, these splits are built based on train data
            - if validation or test splits is missing, the split is built based on train data. The specified
            data split is fixed for all folds.

        Args:
            train_data: training data
            val_data: optional validation data
            test_data: optional test data

        Returns:
            Train, validation and test data fold splits
        """

        file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_key)

        self.folds_path = file_manager.run(filepath=Path(self.prebuilt_folder_name))
        self.folds_path = self.folds_path.joinpath(self.prebuilt_filename)

        return super().run(train_data=train_data,
                           val_data=val_data,
                           test_data=test_data)
