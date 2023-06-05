import abc
from pathlib import Path
from typing import AnyStr, List, Tuple, Any, Union, Optional, cast, Hashable

import numpy as np
import pandas as pd
from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_core.utility.json_utility import save_json, load_json
from sklearn.model_selection import LeaveOneOut

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager
from cinnamon_generic.components.helper import Helper
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor


class Routine(Component):
    """
    A ``Routine`` is a ``Component`` specialized in defining an evaluation routine.
    The ``Routine`` can be viewed as the highest-level ``Component`` since it wraps multiple
    components: from data loading to modeling.
    """

    @abc.abstractmethod
    def build_routine_splits(
            self,
            train_data: Optional[Any] = None,
            val_data: Optional[Any] = None,
            test_data: Optional[Any] = None,
            is_training: bool = False
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """
        Builds dataset splits according to given routine splitting criteria.

        Args:
            train_data: data training split
            val_data: data validation split
            test_data: data test split
            is_training: if True, the ``Routine`` is executing in training mode

        Returns:
            A tuple of train, validation and test splits
        """
        pass

    @abc.abstractmethod
    def routine_step(
            self,
            step_info: FieldDict,
            train_data: Optional[FieldDict] = None,
            val_data: Optional[FieldDict] = None,
            test_data: Optional[FieldDict] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        A ``Routine`` may involve repeating the same execution flow with different input data.
        For instance:
        - Train and test: repeat train and test setting for different random seeds
        - Cross-validation: repeat train and test setting for different inputs and (potentially) different random seeds
        - Leave-one-out: as cross-validation since it is a special case of cross-validation.

        Args:
            step_info:
            train_data: data training split
            val_data: data validation split
            test_data: data test split
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The output of the current ``Routine`` execution flow.
        """

        pass

    @abc.abstractmethod
    def run(
            self,
            helper: Optional[Helper] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``Routine`` component according to its evaluation criteria.

        Args:
            helper: ``Helper`` component for reproducibility and backend interfacing
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``Routine`` output results in ``FieldDict`` format
        """
        pass


class TrainAndTestRoutine(Routine):
    """
    A ``Routine`` extension that implements the 'Train and Test' evaluation routine.
    """

    def get_random_split(
            self,
            data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly splits an input dataset (in pandas.DataFrame format) into two groups.
        In particular, data splitting is done by taking percentage portions of input data
         (via ``validation_percentage`` parameter)

        Args:
            data: data to split in pandas.DataFrame format

        Returns:
            The randomly generated splits.
        """

        amount = int(len(data) * self.validation_percentage)
        all_indexes = np.arange(len(data))
        split_indexes = np.random.choice(all_indexes, size=amount, replace=False)
        remaining_indexes = np.array([idx for idx in all_indexes if idx not in split_indexes])
        split_data = data[split_indexes]
        remaining_data = data[remaining_indexes]
        return remaining_data, split_data

    def build_routine_splits(
            self,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None,
            is_training: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Builds dataset splits according to given routine splitting criteria.

        Args:
            train_data: data training split in pandas.DataFrame format
            val_data: data validation split in pandas.DataFrame format
            test_data: data test split in pandas.DataFrame format
            is_training: if True, the ``Routine`` is executing in training mode

        Returns:
            A tuple of train, validation and test splits in pandas.DataFrame format
        """

        logging_utility.logger.info(f'''
        Building routine splits...
        Train data: {len(train_data) if train_data is not None else train_data}
        Validation data: {len(val_data) if val_data is not None else val_data}
        Test data: {len(test_data) if test_data is not None else test_data}
        ''')

        if is_training:
            if train_data is None:
                raise RuntimeError(f'Training data should be given when training. Got {train_data}')

        if self.has_val_split:
            if train_data is not None and val_data is None:
                if self.validation_percentage is None:
                    raise AttributeError("Routine is expected to build the validation data, "
                                         "but no validation percentage was given")

                logging_utility.logger.info(f'Randomly splitting train data into train and validation splits. '
                                            f'Validation percentage: {self.validation_percentage}')
                train_data, val_data = self.get_random_split(data=train_data)

        if test_data is None and self.parameter_dict["has_test_split"]:
            logging_utility.logger.info(f'Randomly splitting train data into train and test splits. '
                                        f'Test percentage: {self.validation_percentage}')
            train_data, test_data = self.get_random_split(data=train_data)

        logging_utility.logger.info(f'''
        Done!
        Train data: {len(train_data) if train_data is not None else train_data}
        Validation data: {len(val_data) if val_data is not None else val_data}
        Test data: {len(test_data) if test_data is not None else test_data}
        ''')
        return train_data, val_data, test_data

    def routine_step(
            self,
            step_info: FieldDict,
            train_data: Optional[FieldDict] = None,
            val_data: Optional[FieldDict] = None,
            test_data: Optional[FieldDict] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ):
        """
        A single train and test execution flow.

        Args:
            step_info:
            train_data: data training split
            val_data: data validation split
            test_data: data test split
            is_training: if True, the ``TrainAndTestRoutine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The output of the current ``TrainAndTestRoutine`` execution flow.
        """

        serialization_path = Path(serialization_path) \
            if type(serialization_path) != Path and serialization_path is not None \
            else serialization_path

        routine_name = '_'.join([f'{key}-{value}' for key, value in step_info.to_value_dict().items()])
        routine_path = serialization_path.joinpath(routine_name) if serialization_path is not None else None
        if routine_path is not None and not routine_path.is_dir():
            routine_path.mkdir()

        if 'seed' not in step_info:
            raise RuntimeError(f'{self.__class__.__name__} expects a seed for each iteration!')

        self.helper.run(seed=step_info.seed)

        # Pre-Processor
        pre_processor = Processor.build_component_from_key(config_registration_key=self.pre_processor)

        train_data = pre_processor.run(data=train_data,
                                       is_training_data=is_training)
        val_data = pre_processor.run(data=val_data)
        test_data = pre_processor.run(data=test_data)

        # Model
        if self.callbacks is not None:
            callbacks = Callback.build_component_from_key(config_registration_key=self.callbacks)
        else:
            callbacks = None

        model = Model.build_component_from_key(config_registration_key=self.model)
        model.build_model(processor=pre_processor,
                          callbacks=callbacks)

        # Training
        if self.metrics is not None:
            metrics = Metric.build_component_from_key(config_registration_key=self.metrics)
        else:
            metrics = None

        if is_training:
            model.prepare_for_training(train_data=train_data)

            # Model building might require seed re-fixing
            self.helper.run(seed=step_info.seed)

            fit_info = model.fit(train_data=train_data,
                                 val_data=val_data,
                                 metrics=metrics,
                                 callbacks=callbacks)
            step_info.add_short(name='fit_info',
                                value=fit_info,
                                tags={'training'})
        else:
            model.prepare_for_loading(data=test_data if test_data is not None else val_data)

            model.load(serialization_path=routine_path)

            model.check_after_loading()

            # Model loading might require seed re-fixing
            self.helper.run(seed=step_info.seed)

        # Post-Processor
        post_processor: Optional[Processor] = None
        if self.post_processor is not None:
            post_processor = Processor.build_component_from_key(config_registration_key=self.post_processor)

        # Evaluator
        if val_data is not None:
            val_info = model.predict(data=val_data,
                                     metrics=metrics,
                                     callbacks=callbacks)
            if self.post_processor is not None:
                val_info = post_processor.run(data=val_info)
            step_info.add_short(name='val_info',
                                value=val_info,
                                tags={'info'})

        if test_data is not None:
            test_info = model.predict(data=test_data,
                                      metrics=metrics,
                                      callbacks=callbacks)
            if self.post_processor is not None:
                test_info = post_processor.run(data=test_info)
            step_info.add_short(name='test_info',
                                value=test_info,
                                tags={'info'})

        # Save
        if serialization_path is not None:
            model.save(serialization_path=routine_path)

        self.helper.clear_status()

        return step_info

    def run(
            self,
            helper: Optional[Helper] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``TrainAndTestRoutine`` component according to its evaluation criteria.

        Args:
            helper: ``Helper`` component for reproducibility and backend interfacing
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``TrainAndTestRoutine`` output results in ``FieldDict`` format
        """

        routine_result = FieldDict()
        routine_result.add_short(name='steps',
                                 value=[],
                                 type_hint=List[FieldDict])

        # Helper
        if helper is not None:
            self.helper = helper

        seeds = self.seeds if type(self.seeds) == list else [self.seeds]
        self.helper.run(seed=seeds[0])

        # Get data splits
        data_loader = cast(DataLoader, self.data_loader)
        train_data, val_data, test_data = data_loader.get_splits()
        train_data, val_data, test_data = self.build_routine_splits(train_data=train_data,
                                                                    val_data=val_data,
                                                                    test_data=test_data,
                                                                    is_training=is_training)

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)})')

            seed_train_data = data_loader.parse(data=train_data)
            seed_val_data = data_loader.parse(data=val_data)
            seed_test_data = data_loader.parse(data=test_data)

            step_info = FieldDict()
            step_info.add_short(name='seed',
                                value=seed,
                                type_hint=int,
                                tags={'routine_suffix'})
            step_info = self.routine_step(step_info=step_info,
                                          train_data=seed_train_data,
                                          val_data=seed_val_data,
                                          test_data=seed_test_data,
                                          is_training=is_training,
                                          serialization_path=serialization_path)
            routine_result.steps.append(step_info)

        if self.routine_processor is not None:
            routine_result = self.routine_processor.run(data=routine_result)

        return routine_result


class CVRoutine(TrainAndTestRoutine):
    """
    A ``TrainAndTestRoutine`` extension that implements the 'cross-validation' evaluation routine.
    """

    def build_routine_splits(
            self,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None,
            is_training: bool = False,
            key: Optional[str] = None,
            test_indexes: Optional[Union[List[int], np.ndarray]] = None,
            val_indexes: Optional[Union[List[int], np.ndarray]] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Builds dataset splits according to given routine splitting criteria.

        Args:
            train_data: data training split in pandas.DataFrame format
            val_data: data validation split in pandas.DataFrame format
            test_data: data test split in pandas.DataFrame format
            is_training: if True, the ``Routine`` is executing in training mode
            key: column in the given input pandas.DataFrame upon which performing data splitting
            test_indexes: test split indexes given by the cross-validation splitter
            val_indexes: validation split indexes given by the cross-validation splitter

        Returns:
            A tuple of train, validation and test splits in pandas.DataFrame format
        """

        if is_training:
            if key is None:
                raise AttributeError(f'Cannot build data splits without key. Got {key}')

            if test_indexes is None and val_indexes is None:
                raise AttributeError(f'At least validation or test indexes should be given to compute data splits. '
                                     f'Got validation indexes = {val_indexes} and test indexes = {test_indexes}')

            if train_data is None:
                raise RuntimeError(f'Cannot build data splits without training data in training mode. Got {train_data}')

        # Here, train_data might contain: (train, val) or (train, test) according to split keys
        # If (train, test) -> we need to build the validation data

        if train_data is not None:
            # Test data is given -> we use fold keys to define the validation data
            if test_data is not None:
                val_data = train_data[train_data[key].isin(val_indexes).values]
                train_data = train_data[np.logical_not(train_data[key].isin(val_indexes).values)]

            # Test data must be built from fold keys
            else:
                test_data = train_data[train_data[key].isin(test_indexes).values]
                train_data = train_data[~train_data[key].isin(test_indexes).values]

                # We then build the validation data
                # We apply a simple split
                if val_data is None:
                    if val_indexes is None:
                        if self.validation_percentage is None:
                            raise AttributeError(f'Expected a non-null validation percentage. '
                                                 f'Got {self.validation_percentage}')
                        train_data, val_data = self.get_random_split(data=train_data)

                    # The CV split also provides validation indexes
                    else:
                        val_data = train_data[train_data[key].isin(val_indexes).values]
                        train_data = train_data[~train_data[key].isin(val_indexes).values]

        return train_data, val_data, test_data

    def run(
            self,
            helper: Optional[Helper] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``CVRoutine`` component according to its evaluation criteria.

        Args:
            helper: ``Helper`` component for reproducibility and backend interfacing
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``CVRoutine`` output results in ``FieldDict`` format
        """

        routine_result = FieldDict()
        routine_result.add_short(name='steps',
                                 value=[],
                                 type_hint=List[FieldDict])

        # Helper
        if helper is not None:
            self.helper = helper

        seeds = self.seeds if type(self.seeds) == list else [self.seeds]
        self.helper.run(seed=seeds[0])

        # Get data splits
        data_loader = cast(DataLoader, self.data_loader)
        train_data, val_data, test_data = data_loader.get_splits()

        if is_training:
            if train_data is None:
                raise RuntimeError(f'Cannot build cross-validation folds without training data in training mode. '
                                   f'Got {train_data}')

        self.cv_splitter.build_folds(data=train_data,
                                     split_key=self.split_key)

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)}')
            for fold_idx, (train_indexes, val_indexes, test_indexes) in enumerate(self.cv_splitter.run(None)):
                logging_utility.logger.info(f'Fold {fold_idx + 1}/{self.cv_splitter.n_splits}')

                fold_train_data, fold_val_data, fold_test_data = self.build_routine_splits(train_data=train_data,
                                                                                           val_data=val_data,
                                                                                           test_data=test_data,
                                                                                           is_training=is_training,
                                                                                           test_indexes=test_indexes,
                                                                                           key=self.split_key,
                                                                                           val_indexes=val_indexes)
                fold_train_data = data_loader.parse(data=fold_train_data)
                fold_val_data = data_loader.parse(data=fold_val_data)
                fold_test_data = data_loader.parse(data=fold_test_data)

                step_info = FieldDict()
                step_info.add_short(name='seed',
                                    value=seed,
                                    type_hint=int,
                                    tags={'routine_suffix'})
                step_info.add_short(name='fold',
                                    value=fold_idx,
                                    type_hint=int,
                                    tags={'routine_suffix'})
                step_info = self.routine_step(step_info=step_info,
                                              train_data=fold_train_data,
                                              val_data=fold_val_data,
                                              test_data=fold_test_data,
                                              is_training=is_training,
                                              serialization_path=serialization_path)
                routine_result.steps.append(step_info)

        if self.routine_processor is not None:
            routine_result = self.routine_processor.run(data=routine_result)

        return routine_result


class LOORoutine(TrainAndTestRoutine):
    """
    A ``TrainAndTestRoutine`` extension that implements the 'leave-one-out' evaluation routine.
    """

    def build_routine_splits(
            self,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None,
            is_training: bool = False,
            key_values: Optional[Union[List[Any], np.ndarray]] = None,
            key: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Builds dataset splits according to given routine splitting criteria.

        Args:
            train_data: data training split in pandas.DataFrame format
            val_data: data validation split in pandas.DataFrame format
            test_data: data test split in pandas.DataFrame format
            is_training: if True, the ``Routine`` is executing in training mode
            key: column in the given input pandas.DataFrame upon which performing data splitting
            key_values: values corresponding to ``key`` column to keep (the remaining one(s) are left out to
            define the test split)

        Returns:
            A tuple of train, validation and test splits in pandas.DataFrame format
        """

        if is_training:
            if key is None:
                raise AttributeError(f'Cannot build data splits without key. Got {key}')

            if key_values is None:
                raise AttributeError(f'Non-left out {key} values should be given to compute LOO splits. '
                                     f'Got {key_values}')

            if train_data is None:
                raise RuntimeError(f'Cannot build data splits without training data in training mode. Got {train_data}')

        if train_data is not None:
            # We have validation data -> we build test data according to split keys
            if val_data is not None:
                test_data = train_data[train_data[key].isin(key_values).values]
                train_data = train_data[~train_data[key].isin(key_values).values]

            # No validation data
            else:
                # If we don't have test data -> we build it according to split key
                if test_data is None:
                    test_data = train_data[train_data[key].isin(key_values).values]
                    train_data = train_data[~train_data[key].isin(key_values).values]

                    # Then we have to build the validation data
                    if self.validation_percentage is None:
                        raise AttributeError(f'Cannot compute train and validation splits via random splitting without '
                                             f'specifying the validation amount. '
                                             f'Got validation percentage = {self.validation_percentage}')
                    train_data, val_data = self.get_random_split(data=train_data)

                # We might not have test data (this is fine, if not explicitly requested)
                else:
                    val_data = train_data[train_data[key].isin(key_values).values]
                    train_data = train_data[~train_data[key].isin(key_values).values]

        return train_data, val_data, test_data

    def run(
            self,
            helper: Optional[Helper] = None,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``LOORoutine`` component according to its evaluation criteria.

        Args:
            helper: ``Helper`` component for reproducibility and backend interfacing
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``LOORoutine`` output results in ``FieldDict`` format
        """

        routine_result = FieldDict()
        routine_result.add_short(name='steps',
                                 value=[],
                                 type_hint=List[FieldDict])

        # Helper
        if helper is not None:
            self.helper = helper

        seeds = self.seeds if type(self.seeds) == list else [self.seeds]
        self.helper.run(seed=seeds[0])

        # Get data splits
        data_loader = cast(DataLoader, self.data_loader)
        train_data, val_data, test_data = data_loader.get_splits()

        if is_training:
            if train_data is None:
                raise RuntimeError(f'Cannot execute in training mode without training data.'
                                   f' Got train_data={train_data}')

        self.cv_splitter.build_split_values(data=train_data,
                                            split_key=self.split_key)

        split_values = self.cv_splitter.split_values

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)}')
            for train_keys_indexes, excluded_key_indexes in self.cv_splitter.run():
                excluded_key = split_values[excluded_key_indexes]
                logging_utility.logger.info(f'LOO Fold: {excluded_key}')

                fold_train_data, fold_val_data, fold_test_data = self.build_routine_splits(train_data=train_data,
                                                                                           val_data=val_data,
                                                                                           test_data=test_data,
                                                                                           is_training=is_training,
                                                                                           key_values=excluded_key,
                                                                                           key=self.split_key)
                fold_train_data = data_loader.parse(data=fold_train_data)
                fold_val_data = data_loader.parse(data=fold_val_data)
                fold_test_data = data_loader.parse(data=fold_test_data)

                step_info = FieldDict()
                step_info.add_short(name='seed',
                                    value=seed,
                                    type_hint=int,
                                    tags={'routine_suffix'})
                step_info.add_short(name='fold',
                                    value=excluded_key,
                                    type_hint=Hashable,
                                    tags={'routine_suffix'})
                step_info = self.routine_step(step_info=step_info,
                                              train_data=fold_train_data,
                                              val_data=fold_val_data,
                                              test_data=fold_test_data,
                                              is_training=is_training,
                                              serialization_path=serialization_path)
                routine_result.steps.append(step_info)

        if self.routine_processor is not None:
            routine_result = self.routine_processor.run(data=routine_result)

        return routine_result


class CVSplitter(Component):

    @abc.abstractmethod
    def build_folds(
            self,
            data: pd.DataFrame,
            split_key: Hashable
    ):
        pass

    @abc.abstractmethod
    def build_all_splits_folds(
            self,
            X: Any,
            y: Any,
            validation_n_splits: Optional[int] = None
    ):
        pass

    @abc.abstractmethod
    def split(
            self,
            X: Any,
            y: Optional[Any] = None,
            groups: Optional[Any] = None
    ):
        pass


class PrebuiltCVSplitter(CVSplitter):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.folds = None
        self.key_listing = None
        self.cv = self.cv_type(n_splits=self.n_splits, shuffle=self.shuffle)

        file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_registration_key)

        self.folds_path = file_manager.run(filepath=Path(self.prebuilt_folder))
        self.folds_path = self.folds_path.joinpath(self.prebuilt_filename)

    def _build_folds(
            self,
            data: pd.DataFrame,
            split_key: Hashable
    ):
        """
        Builds and stores cross-validation folds from input pandas.DataFrame.
        Folds are built by using the internal CV splitter class from scikit-learn.

        Args:
            data: input data from which to build splits.
            split_key: the data field to consider for building splits.
        """

        data_labels = data[split_key].values
        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(data, data_labels)):
            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes,
                self.held_out_key: held_out_indexes
            }

    def build_folds(
            self,
            data: pd.DataFrame,
            split_key: Hashable,
    ):
        """
        Builds cross-validation folds from input pandas.DataFrame or loads pre-built ones from filesystem.

        Args:
            data: input data from which to build splits.
            split_key: the data field to consider for building splits.
        """
        if self.folds_path is not None and self.folds_path.exists():
            self.load_folds()
        else:
            self._build_folds(data=data,
                              split_key=split_key)

    def build_all_splits_folds(
            self,
            X: Any,
            y: Any,
            validation_n_splits: Optional[int] = None
    ):
        """
        Builds train, validation and test split folds from input data using the internal CV splitter.

        Args:
            X: input data
            y: input ground-truth labels
            validation_n_splits: number of splits for validation set
        """
        if self.held_out_key != 'test':
            raise AttributeError(f'Expected held_out_key to be equal to "test" but got {self.held_out_key}')

        validation_n_splits = self.n_splits if validation_n_splits is None else validation_n_splits

        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(X, y)):
            sub_X = X[train_indexes]
            sub_y = y[train_indexes]

            self.cv.n_splits = validation_n_splits
            sub_train_indexes, sub_val_indexes = list(self.cv.split(sub_X, sub_y))[0]
            self.cv.n_splits = self.n_splits

            self.folds[f'fold_{fold}'] = {
                'train': train_indexes[sub_train_indexes],
                self.held_out_key: held_out_indexes,
                'validation': train_indexes[sub_val_indexes]
            }

    def save_folds(
            self,
    ):
        """
        Saves built cross-validation folds to filesystem for quick re-use in JSON format.

        Args:
            save_path: path where to save built cross-validation folds
        """
        save_json(self.folds_path, self.folds)

    def load_folds(
            self
    ):
        """
        Loads pre-built cross-validation folds from filesystem.

        Args:
            load_path: path where to load pre-built cross-validation folds
        """

        self.folds = load_json(self.folds_path)
        self.n_splits = len(self.folds)
        key_path = self.folds_path.parent / (self.folds_path.stem + '_listing.json')
        self.key_listing = load_json(key_path)
        self.key_listing = np.array(self.key_listing)

    def _iter_test_indexes(
            self,
            X: Optional[Any] = None,
            y: Optional[Any] = None,
            groups: Optional[Any] = None
    ):
        """
        Generates integer indexes corresponding to test sets.

        Args:
            X: Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            y: The target variable for supervised learning problems.

            groups: Group labels for the samples used while splitting the dataset into train/test set.
        """

        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            yield self.folds[fold][self.held_out_key]

    def split(
            self,
            X: Any,
            y: Optional[Any] = None,
            groups: Optional[Any] = None
    ):
        """Generate indexes to split data into training and test set.

        Args:
            X: Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            y: The target variable for supervised learning problems.

            groups: Group labels for the samples used while splitting the dataset into train/validation/test set.

        Yields:
            train: The training set indexes for that split.

            val: The validation set indexes for that split if ``return_val_indexes``

            test: The testing set indexes for that split.
        """
        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            val_indexes = self.key_listing[self.folds[fold]['validation']] if 'validation' in self.folds[fold] else None
            test_indexes = self.key_listing[self.folds[fold]['test']] if 'test' in self.folds[fold] else None

            assert val_indexes is not None or test_indexes is not None

            train_indexes = self.key_listing[self.folds[fold]['train']]

            yield train_indexes, val_indexes, test_indexes

    def run(
            self,
            X: Any,
            y: Optional[Any] = None,
            groups: Optional[Any] = None
    ) -> Any:
        return self.split(X=X, y=y, groups=groups)


class LOOSplitter(Component):

    @abc.abstractmethod
    def build_split_values(
            self,
            data: pd.DataFrame,
            split_key: Hashable
    ):
        """
        Retrieves unique values for specified ``split_key`` in input ``train_data``.
        The retrieved values define the leave-one-out groups.

        Args:
            data: input data from which to define leave-one-out folds
            split_key: data field to use to build folds.
        """
        pass


class PrebuiltLOOSplitter(LOOSplitter):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.split_values = None
        self.loo = LeaveOneOut()

        file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_registration_key)

        self.folds_path = file_manager.run(filepath=Path(self.prebuilt_folder))
        self.folds_path = self.folds_path.joinpath(self.prebuilt_filename)

    def build_split_values(
            self,
            data: pd.DataFrame,
            split_key: Hashable
    ):
        """
        Retrieves unique values for specified ``split_key`` in input ``train_data``.
        The retrieved values define the leave-one-out groups.

        Args:
            data: input data from which to define leave-one-out folds
            split_key: data field to use to build folds.
        """
        if not self.folds_path.exists():
            self.split_values = np.unique(data[split_key].values)
            self.save_split_values()
        else:
            self.load_split_values()

    def save_split_values(
            self
    ):
        """
        Serializes retrieved leave-one-out groups to filesystem in JSON format for quick re-use.
        """

        if self.folds_path is None:
            raise RuntimeError(f'Expected a non-null ``folds_path``. Got {self.folds_path}')
        save_json(self.folds_path, self.split_values)

    def load_split_values(
            self
    ):
        """
        Loads serialized leave-one-out groups from the filesystem.
        """

        if self.folds_path is None:
            raise RuntimeError(f'Expected a non-null ``folds_path``. Got {self.folds_path}')
        self.split_values = load_json(self.folds_path)
