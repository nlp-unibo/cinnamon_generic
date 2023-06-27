import abc
from pathlib import Path
from typing import Tuple, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from cinnamon_core.core.component import Component
from cinnamon_core.utility.pickle_utility import save_pickle, load_pickle
from cinnamon_generic.components.file_manager import FileManager


class InternalTTSplitter:

    @abc.abstractmethod
    def split(
            self,
            data: Any,
            size: int
    ) -> Tuple[Any, Any]:
        pass


class SklearnTTSplitter(InternalTTSplitter):

    def __init__(
            self,
            **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def split(
            self,
            data: Any,
            size: int
    ) -> Tuple[Any, Any]:
        return train_test_split(data,
                                test_size=size,
                                random_state=getattr(self, 'random_state') if hasattr(self, 'random_state') else None,
                                shuffle=getattr(self, 'shuffle') if hasattr(self, 'shuffle') else None,
                                stratify=getattr(self, 'stratify') if hasattr(self, 'stratify') else None)


class TTSplitter(Component):

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

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_key)

        self.folds_path = file_manager.run(filepath=Path(self.prebuilt_folder))
        self.folds_path = self.folds_path.joinpath(self.prebuilt_filename)

        self.folds = []

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
