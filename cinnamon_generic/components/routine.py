import abc
from pathlib import Path
from typing import AnyStr, List, Union, Optional, cast

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.data_loader import DataLoader
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
        model.build(processor=pre_processor,
                    callbacks=callbacks)

        if callbacks is not None:
            callbacks.setup(component=model,
                            save_path=serialization_path)

        # Training
        if self.metrics is not None:
            metrics = Metric.build_component_from_key(config_registration_key=self.metrics)
        else:
            metrics = None

        # Model Processor
        model_processor: Optional[Processor] = None
        if self.model_processor is not None:
            model_processor = Processor.build_component_from_key(config_registration_key=self.model_processor)

        if is_training:
            model.prepare_for_training(train_data=train_data)

            # Model building might require seed re-fixing
            self.helper.run(seed=step_info.seed)

            fit_info = model.fit(train_data=train_data,
                                 val_data=val_data,
                                 metrics=metrics,
                                 callbacks=callbacks,
                                 model_processor=model_processor)
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
                                     callbacks=callbacks,
                                     model_processor=model_processor)
            if self.post_processor is not None:
                val_info = post_processor.run(data=val_info)
            step_info.add_short(name='val_info',
                                value=val_info,
                                tags={'info'})

        if test_data is not None:
            test_info = model.predict(data=test_data,
                                      metrics=metrics,
                                      callbacks=callbacks,
                                      model_processor=model_processor)
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

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)})')

            train_data, val_data, test_data = data_loader.get_splits()
            step_train_data, step_val_data, step_test_data = self.data_splitter.run(train_data=train_data,
                                                                                    val_data=val_data,
                                                                                    test_data=test_data)

            step_train_data = data_loader.parse(data=step_train_data)
            step_val_data = data_loader.parse(data=step_val_data)
            step_test_data = data_loader.parse(data=step_test_data)

            step_info = FieldDict()
            step_info.add_short(name='seed',
                                value=seed,
                                type_hint=int,
                                tags={'routine_suffix'})
            step_info = self.routine_step(step_info=step_info,
                                          train_data=step_train_data,
                                          val_data=step_val_data,
                                          test_data=step_test_data,
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
        train_data, val_data, test_data = self.data_loader.get_splits()

        if is_training:
            if train_data is None:
                raise RuntimeError(f'Cannot build cross-validation folds without training data in training mode. '
                                   f'Got {train_data}')

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)})')
            for fold_idx, \
                    (step_train_data,
                     step_val_data,
                     step_test_data) in enumerate(self.data_splitter.run(train_data=train_data,
                                                                         val_data=val_data,
                                                                         test_data=test_data)):
                logging_utility.logger.info(f'Fold {fold_idx + 1}')

                step_train_data = self.data_loader.parse(data=step_train_data)
                step_val_data = self.data_loader.parse(data=step_val_data)
                step_test_data = self.data_loader.parse(data=step_test_data)

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
                                              train_data=step_train_data,
                                              val_data=step_val_data,
                                              test_data=step_test_data,
                                              is_training=is_training,
                                              serialization_path=serialization_path)
                routine_result.steps.append(step_info)

        if self.routine_processor is not None:
            routine_result = self.routine_processor.run(data=routine_result)

        return routine_result
