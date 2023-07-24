import pickle
import shutil
import time
from enum import Enum
from multiprocessing import Process, Pool, Queue
from subprocess import Popen
from typing import Dict, Any, Optional

import hyperopt
import numpy as np
from hyperopt import fmin, space_eval, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_core.utility.pickle_utility import save_pickle
from cinnamon_core.utility.python_utility import get_dict_values_combinations
from cinnamon_generic.components.file_manager import FileManager
from pathlib import Path


class ValidateCondition(str, Enum):
    MAXIMIZATION = 'maximization'
    MINIMIZATION = 'minimization'


class MissingSearchSpaceException(Exception):

    def __init__(self, search_space):
        super().__init__(f'Cannot execute calibration without search space! Got {search_space}')


class DBNotSpecifiedException(Exception):

    def __init__(self, db_name):
        super().__init__(f'Cannot execute calibration without specifying a db name! Got {db_name}')


class UnsetValidatorException(Exception):

    def __init__(self):
        super().__init__('Validator is not set! Make sure to run set_validator() before calibration')


# TODO: add OptunaCalibrator
class Calibrator(Component):
    """
    A ``Calibrator`` is a ``Component`` specialized in hyper-parameters tuning.
    """

    def evaluate_combination(
            self,
            parameter_combination: Dict[str, Any]
    ):
        """
        Evaluation method of each ``validator`` delta copy based on sampled hyper-parameter combinations.
        Given an input hyper-parameter combination (``parameter_combination``):
        - A ``validator`` delta copy is built
        - The ``validator`` delta copy is run
        - The ``validator`` result is inspected to get the value of specified calibration metric

        The ``Calibrator`` always executes my minimizing a certain metric. Thus, for metrics that have to maximized,
        their negative value is computed.

        Args:
            parameter_combination: a sampled hyper-parameter combination of ``validator``'s ``Configuration``.

        Returns:
            The calibration metric value for current ``validator``'s delta copy
            according to sampled hyper-parameter combination.
        """

        logging_utility.logger.info(f'Considering hyper-parameters: {parameter_combination}')

        # Create a validator copy with passed arguments
        validator = self.validator.get_delta_copy(params_dict=parameter_combination)

        # Run validator
        validator_results = validator.run(**self.validator_args)

        # Get validation results
        validation_on_value: float = validator_results[self.validate_on]
        if validation_on_value is None or type(validation_on_value) != float:
            raise RuntimeError(
                f'Expected a float validation value for {self.validate_on} '
                f'but got {validation_on_value} (type={type(validation_on_value)}).'
                f'Are you sure you {self.validate_on} is correct?')

        if self.validate_condition == ValidateCondition.MAXIMIZATION:
            return -validation_on_value

        return validation_on_value


# TODO: make multiprocessing
# TODO: use a DB or file to store/load progress
class GridSearchCalibrator(Calibrator):
    """
    A ``Calibrator`` extension that defines a grid-search calibration criteria.
    """

    def run(
            self,
    ) -> Any:
        """
        Runs the calibration phase for the specified ``validator`` ``Component``.
        The grid-search samples and evaluates all possible hyper-parameter combinations of the
        ``validator``'s ``Configuration``.

        Returns:
            The best hyper-parameter combination along with the corresponding calibration metric value.
        """

        if self.validator is None:
            raise UnsetValidatorException()

        search_space = self.validator.config.get_search_space()

        if search_space is None:
            raise MissingSearchSpaceException(search_space=search_space)

        combinations = get_dict_values_combinations(search_space)

        logging_utility.logger.info(
            f'Starting hyper-parameters calibration search! Total combinations: {len(combinations)}')

        calibration_results = []
        for combination in tqdm(combinations):
            combination_result = self.evaluate_combination(parameter_combination=combination)
            calibration_results.append((combination_result, combination))

        calibration_results = sorted(calibration_results, key=lambda pair: pair[0])
        best_result, best_params = calibration_results[0]

        return FieldDict({"best_result": best_result,
                          "best_params": best_params,
                          'combinations': len(combinations),
                          'calibration_results': calibration_results})


class RandomSearchCalibration(Calibrator):
    """
    A ``Calibrator`` extension that defines a random search calibration criteria.
    """

    def run(
            self,
    ) -> Any:
        """
        Runs the calibration phase for the specified ``validator`` ``Component``.
        The grid-search samples and evaluates all possible hyper-parameter combinations of the
        ``validator``'s ``Configuration``.

        Returns:
            The best hyper-parameter combination along with the corresponding calibration metric value.
        """

        if self.validator is None:
            raise UnsetValidatorException()

        search_space = self.validator.config.get_search_space()

        if search_space is None:
            raise MissingSearchSpaceException(search_space=search_space)

        combinations = get_dict_values_combinations(search_space)

        self.tries = min(self.tries, len(combinations))

        logging_utility.logger.info(f'Starting hyper-parameters calibration search! '
                                    f'Total combinations: {len(combinations)} '
                                    f'Total tries: {self.tries}')

        calibration_results = []
        sampled_combinations = np.random.choice(combinations, size=self.tries, replace=False)
        for combination in tqdm(sampled_combinations):
            combination_result = self.evaluate_combination(parameter_combination=combination)
            calibration_results.append((combination_result, combination))

        calibration_results = list(sorted(calibration_results, key=lambda pair: pair[0]))
        best_result, best_params = calibration_results[0]

        return FieldDict({"best_result": best_result,
                          "best_params": best_params,
                          'combinations': len(combinations),
                          'calibration_results': calibration_results})


class HyperOptCalibrator(Calibrator):
    """
    A ``Calibrator`` extension that wraps a ``HyperOpt`` calibrator.
    """

    def __init__(
            self,
            **kwargs
    ):
        """
        The ``HyperOptCalibrator`` constructor.
        The ``FileManager`` component is used to initialize MongoDB-related paths (if used).
        """

        super().__init__(**kwargs)

        self.mongo_directory: Optional[Path] = None
        self.mongo_workers_directory: Optional[Path] = None

    def _retrieve_custom_trials(
            self,
            db_name: str
    ):
        """
        Retrieves an existing HyperOpt ``Trials`` instance (if it exists).

        Args:
            db_name: name of the MongoDB database.

        Returns:
            The retrieved ``Trials`` instance.
        """

        if not self.mongo_directory.is_dir():
            raise FileNotFoundError(f'Mongo directory is not a directory: {self.mongo_directory}')

        trials_path = self.mongo_directory.joinpath(f'{db_name}.pickle')
        if trials_path.exists():
            logging_utility.logger.info('Using existing Trials DB!')
            with trials_path.open('rb') as f:
                trials = pickle.load(f)
        else:
            logging_utility.logger.info(f"Can't find specified Trials DB ({db_name})...creating new one!")
            trials = Trials(exp_key=db_name)
        return trials

    def _calibrate(
            self,
            trials: Trials,
            search_space: Any
    ):
        """
        Runs HyperOpt calibration.

        Args:
            trials: HyperOpt ``Trials`` instance that keeps track of evaluated hyper-parameter combinations.
            search_space: hyper-parameter search space defined via HyperOpt operations.

        Returns:
            The best HyperOpt trial
        """

        best = fmin(self.evaluate_combination,
                    search_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evaluations,
                    trials=trials,
                    show_progressbar=False,
                    **self.hyperopt_additional_info)
        return best

    def _mongo_calibrate(
            self,
            trials: Trials,
            search_space: Any,
            queue: Queue
    ):
        """
        Runs HyperOpt calibration for MongoDB-based execution.

        Args:
            trials: HyperOpt ``Trials`` instance that keeps track of evaluated hyper-parameter combinations.
            search_space: hyper-parameter search space defined via HyperOpt operations.
            queue: queue where to store each mongo worker's best trial.
        """

        best = self._calibrate(trials=trials, search_space=search_space)
        queue.put(best)

    def _run_mongo_worker(
            self,
            db_name: str
    ):
        """
        Executes a HyperOpt-based mongo worker. Each worker samples hyper-parameter combination from the search space
        and evaluates it.

        Args:
            db_name: name of the MongoDB database.
        """

        # Define command
        cmd = ['hyperopt-mongo-worker',
               f'--mongo={self.mongo_address}:{self.mongo_port}/{db_name}',
               f'--poll-interval={self.poll_interval}',
               f'--reserve-timeout={self.reserve_timeout}',
               f'--max-consecutive-failures={self.max_consecutive_failures}']
        if not self.use_subprocesses:
            cmd.append(f'--no-subprocesses')

        cmd = ' '.join(cmd)
        logging_utility.logger.info(f'Executing mongo worker...\n{cmd}')

        # Clear mongo_workers_dir before execution
        if self.mongo_workers_directory.exists():
            shutil.rmtree(self.mongo_workers_directory)

        # Run process
        process = Popen(cmd, cwd=self.mongo_workers_directory, shell=True)
        process.wait()

    def run(
            self,
    ):
        """
        Runs the calibration phase for the specified ``validator`` ``Component``.
        The HyperOpt calibration samples up to ``max_evaluations`` hyper-parameter combinations (i.e., trials).
        Note that the same trial might be sampled multiple times.
        The ``HyperOptCalibrator`` can execute in two different modes:
        - MongoDB mode: if a MongoDB instance is running, the calibrator runs multiple workers to evaluate
        multiple trials in parallel. All results are stored in the MongoDB instance.
        - Sequential mode: trials are evaluated sequentially. All results are stored in a .csv file.

        Returns:
            The best hyper-parameter combination along with the corresponding calibration metric value.
        """

        logging_utility.logger.info(
            f'Starting hyper-parameters calibration search! Max evaluations: {self.max_evaluations}')

        if self.validator is None:
            raise UnsetValidatorException()

        search_space = self.validator.config.get_search_space()

        if search_space is None:
            raise MissingSearchSpaceException(search_space=search_space)

        if self.db_name is None:
            raise DBNotSpecifiedException(db_name=self.db_name)

        file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_key)
        self.mongo_directory = file_manager.run(filepath=self.mongo_directory_name)
        self.mongo_workers_directory = file_manager.run(filepath=self.mongo_workers_directory_name)

        if self.use_mongo:
            trials = MongoTrials(
                f'mongo://{self.mongo_address}:{self.mongo_port}/{self.db_name}/jobs',
                exp_key='exp1')
        else:
            trials = self._retrieve_custom_trials(db_name=self.db_name)

        if self.use_mongo:
            logging_utility.logger.info('Running calibration with mongodb, make sure mongodb is active and running!')

            # Execute main calibration process as subprocess
            main_calibrator_queue = Queue()
            main_calibrator_process = Process(target=self._mongo_calibrate,
                                              args=(trials, search_space, main_calibrator_queue))
            main_calibrator_process.start()

            # Execute workers
            with Pool(processes=self.workers) as pool:

                # Sleep ensures that different folders will be created
                workers_results = []
                for _ in range(self.workers):
                    workers_results.append(pool.apply_async(self._run_mongo_worker, (self.db_name,)))
                    time.sleep(self.worker_sleep_interval)

                workers_results = [worker.get() for worker in workers_results]

            # Wait for main calibration process
            main_calibrator_process.join()
            best = main_calibrator_queue.get()
        else:
            best = self._calibrate(trials=trials, search_space=search_space)

        best_params = space_eval(search_space, best)
        try:
            best_trial = trials.best_trial
        except hyperopt.exceptions.AllTrialsFailed:
            best_trial = {'state': 'N/A'}

        logging_utility.logger.info('Hyper-parameters calibration ended..')
        logging_utility.logger.info(f'Best combination: {best_params}')
        logging_utility.logger.info(f'Best combination info: {best_trial}')

        if not self.use_mongo:
            save_pickle(filepath=self.mongo_directory.joinpath(f'{self.db_name}.pickle'), data=trials)

        return FieldDict({"best_result": best_trial,
                          "best_params": best_params})


__all__ = [
    'ValidateCondition',
    'Calibrator',
    'GridSearchCalibrator',
    'RandomSearchCalibration',
    'HyperOptCalibrator'
]
