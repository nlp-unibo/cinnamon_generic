import abc
import os
from pathlib import Path
from typing import AnyStr, List, Union, Optional

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback
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
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``Routine`` component according to its evaluation criteria.

        Args:
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
        pre_processor = Processor.build_component_from_key(registration_key=self.pre_processor)
        if not is_training:
            pre_processor.load(serialization_path=routine_path)

        logging_utility.logger.info('Pre-processing train data...')
        train_data = pre_processor.run(data=train_data,
                                       is_training_data=is_training)

        logging_utility.logger.info('Pre-processing val data...')
        val_data = pre_processor.run(data=val_data)

        logging_utility.logger.info('Pre-processing test data...')
        test_data = pre_processor.run(data=test_data)

        pre_processor.finalize()

        # Model
        logging_utility.logger.info('Building model...')
        model = Model.build_component_from_key(registration_key=self.model)
        model.build(processor=pre_processor,
                    callbacks=self.callbacks)

        if self.callbacks is not None:
            self.callbacks.setup(component=model,
                                 save_path=serialization_path)

        # Training
        if self.metrics is not None:
            metrics = Metric.build_component_from_key(registration_key=self.metrics)
        else:
            metrics = None

        # Model Processor
        model_processor: Optional[Processor] = None
        if self.model_processor is not None:
            model_processor = Processor.build_component_from_key(registration_key=self.model_processor)

        if is_training:
            model.prepare_for_training(train_data=train_data)

            # Model building might require seed re-fixing
            self.helper.run(seed=step_info.seed)

            fit_info = model.fit(train_data=train_data,
                                 val_data=val_data,
                                 metrics=metrics,
                                 callbacks=self.callbacks,
                                 model_processor=model_processor)
            step_info.add(name='fit_info',
                          value=fit_info,
                          tags={'training'})
        else:
            model.prepare_for_loading(data=test_data if test_data is not None else val_data)

            model.load(serialization_path=routine_path)

            model.check_after_loading()

            # Model loading might require seed re-fixing
            self.helper.run(seed=step_info.seed)

        routine_suffixes = step_info.search_by_tag('routine_suffix')
        train_info = model.evaluate(data=train_data,
                                    metrics=metrics,
                                    callbacks=self.callbacks,
                                    model_processor=model_processor,
                                    suffixes={**routine_suffixes, **{'split': 'train', 'status': 'inference'}})
        step_info.add(name='train_info',
                      value=train_info,
                      tags={'info'})

        # Evaluator
        if val_data is not None:
            val_info = model.evaluate(data=val_data,
                                      metrics=metrics,
                                      callbacks=self.callbacks,
                                      model_processor=model_processor,
                                      suffixes={**routine_suffixes, **{'split': 'val', 'status': 'inference'}})
            step_info.add(name='val_info',
                          value=val_info,
                          tags={'info'})

        if test_data is not None:
            test_info = model.evaluate(data=test_data,
                                       metrics=metrics,
                                       callbacks=self.callbacks,
                                       model_processor=model_processor,
                                       suffixes={**routine_suffixes, **{'split': 'test', 'status': 'inference'}})
            step_info.add(name='test_info',
                          value=test_info,
                          tags={'info'})

        if model_processor is not None:
            model_processor.finalize()

        # Post-Processor
        if self.post_processor is not None:
            post_processor = Processor.build_component_from_key(registration_key=self.post_processor)
            step_info = post_processor.run(data=step_info)
            post_processor.finalize()

        # Save
        if serialization_path is not None:
            pre_processor.save(serialization_path=routine_path)
            model.save(serialization_path=routine_path)

        self.helper.clear_status()

        return step_info

    def run(
            self,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``TrainAndTestRoutine`` component according to its evaluation criteria.

        Args:
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``TrainAndTestRoutine`` output results in ``FieldDict`` format
        """

        if self.callbacks is not None:
            self.callbacks = Callback.build_component_from_key(registration_key=self.callbacks)

        routine_info = FieldDict()
        routine_info.add(name='steps',
                         value=[],
                         type_hint=List[FieldDict])

        # Helper
        seeds = self.seeds if type(self.seeds) == list else [self.seeds]
        self.helper.run(seed=seeds[0])

        if self.callbacks is not None:
            self.callbacks.run(hookpoint='on_routine_begin',
                               logs={
                                   'seeds': seeds
                               })

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)})')

            if self.callbacks is not None:
                self.callbacks.run(hookpoint='on_routine_step_begin',
                                   logs={
                                       'seed': seed,
                                       'serialization_path': serialization_path,
                                       'is_training': is_training
                                   })

            train_data, val_data, test_data = self.data_loader.get_splits()
            step_train_data, step_val_data, step_test_data = self.data_splitter.run(train_data=train_data,
                                                                                    val_data=val_data,
                                                                                    test_data=test_data)

            step_train_data = self.data_loader.parse(data=step_train_data)
            step_val_data = self.data_loader.parse(data=step_val_data)
            step_test_data = self.data_loader.parse(data=step_test_data)

            step_info = FieldDict()
            step_info.add(name='seed',
                          value=seed,
                          type_hint=int,
                          tags={'routine_suffix'})
            step_info = self.routine_step(step_info=step_info,
                                          train_data=step_train_data,
                                          val_data=step_val_data,
                                          test_data=step_test_data,
                                          is_training=is_training,
                                          serialization_path=serialization_path)
            routine_info.steps.append(step_info)

            if self.callbacks is not None:
                self.callbacks.run(hookpoint='on_routine_step_end',
                                   logs={
                                       'seed': seed,
                                       'step_info': step_info
                                   })

        if self.routine_processor is not None:
            routine_info = self.routine_processor.run(data=routine_info)

        if self.callbacks is not None:
            self.callbacks.run(hookpoint='on_routine_end',
                               logs={
                                   'seeds': seeds,
                                   'routine_info': routine_info,
                                   'serialization_path': serialization_path,
                                   'is_training': is_training
                               })

        return routine_info


class CVRoutine(TrainAndTestRoutine):
    """
    A ``TrainAndTestRoutine`` extension that implements the 'cross-validation' evaluation routine.
    """

    def run(
            self,
            is_training: bool = False,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
    ) -> FieldDict:
        """
        Executes the ``CVRoutine`` component according to its evaluation criteria.

        Args:
            is_training: if True, the ``Routine`` is executing in training mode
            serialization_path: Path where to store any ``Component``'s serialization activity.

        Returns:
            The ``CVRoutine`` output results in ``FieldDict`` format
        """

        if self.callbacks is not None:
            self.callbacks = Callback.build_component_from_key(registration_key=self.callbacks)

        routine_info = FieldDict()
        routine_info.add(name='steps',
                         value=[],
                         type_hint=List[FieldDict])

        # Helper
        seeds = self.seeds if type(self.seeds) == list else [self.seeds]
        self.helper.run(seed=seeds[0])

        # Get data splits
        train_data, val_data, test_data = self.data_loader.get_splits()

        if is_training:
            if train_data is None:
                raise RuntimeError(f'Cannot build cross-validation folds without training data in training mode. '
                                   f'Got {train_data}')

        if self.callbacks is not None:
            self.callbacks.run(hookpoint='on_routine_begin',
                               logs={
                                   'seeds': seeds,
                                   'serialization_path': serialization_path,
                                   'is_training': is_training
                               })

        for seed_idx, seed in enumerate(seeds):
            logging_utility.logger.info(f'Seed: {seed} (Progress -> {seed_idx + 1}/{len(seeds)})')
            for fold_idx, \
                    (step_train_data,
                     step_val_data,
                     step_test_data) in enumerate(self.data_splitter.run(train_data=train_data,
                                                                         val_data=val_data,
                                                                         test_data=test_data)):
                if fold_idx == self.max_folds:
                    break

                logging_utility.logger.info(f'Fold {fold_idx + 1}')

                logging_utility.logger.info(f'Train size: {len(step_train_data)}{os.linesep}'
                                            f'Validation size: {len(step_val_data)}{os.linesep}'
                                            f'Test size: {len(step_test_data)}{os.linesep}')

                if self.callbacks is not None:
                    self.callbacks.run(hookpoint='on_routine_step_begin',
                                       logs={
                                           'seed': seed,
                                           'fold': fold_idx
                                       })

                step_train_data = self.data_loader.parse(data=step_train_data)
                step_val_data = self.data_loader.parse(data=step_val_data)
                step_test_data = self.data_loader.parse(data=step_test_data)

                step_info = FieldDict()
                step_info.add(name='seed',
                              value=seed,
                              type_hint=int,
                              tags={'routine_suffix'})
                step_info.add(name='fold',
                              value=fold_idx,
                              type_hint=int,
                              tags={'routine_suffix'})
                step_info = self.routine_step(step_info=step_info,
                                              train_data=step_train_data,
                                              val_data=step_val_data,
                                              test_data=step_test_data,
                                              is_training=is_training,
                                              serialization_path=serialization_path)
                routine_info.steps.append(step_info)

                if self.callbacks is not None:
                    self.callbacks.run(hookpoint='on_routine_step_end',
                                       logs={
                                           'seed': seed,
                                           'fold': fold_idx,
                                           'step_info': step_info
                                       })

        if self.routine_processor is not None:
            routine_info = self.routine_processor.run(data=routine_info)

        if self.callbacks is not None:
            self.callbacks.run(hookpoint='on_routine_end',
                               logs={
                                   'seeds': seeds,
                                   'routine_info': routine_info,
                                   'serialization_path': serialization_path,
                                   'is_training': is_training
                               })

        return routine_info
