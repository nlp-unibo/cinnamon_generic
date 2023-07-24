from datetime import datetime
from pathlib import Path
from typing import AnyStr, Optional, Union

import pandas as pd
from cinnamon_core.core.component import Component
from cinnamon_core.core.registry import Registration
from cinnamon_core.utility import logging_utility


class FileManager(Component):
    """
    The ``FileManager`` is a ``Component`` that handles file and folder paths.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.base_directory: Optional[Union[AnyStr, Path]] = None
        self.runs_registry = {}

    def setup(
            self,
            base_directory: Union[AnyStr, Path]
    ):
        """
        Initializes the ``FileManager`` with the base directory.
        All file and folder paths of the ``FileManager`` will be paths relative to the base directory.

        Args:
            base_directory: path to base directory for ``FileManager``.
        """

        self.base_directory = base_directory
        directories = [param for param_key, param in self.config.items() if 'directory' in param.tags]
        for directory in directories:
            directory.value = self.run(filepath=directory.value,
                                       create_path=False)

    def register_temporary_run_name(
            self,
            key: Registration,
            replacement_name: Optional[str] = None,
            create_path: bool = False
    ) -> Path:
        """
        Internally stores a temporary run name, which can be optionally overriden.

        Args:
            key: configuration registration key
            replacement_name: replacement name for temporary run name
            create_path: if True, a folder with temporary run name is created

        Returns:
            The temporary run path
        """

        current_date = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        run_path = self.runs_directory.joinpath(key.namespace, key.name, current_date)

        if replacement_name is not None:
            self.runs_registry[run_path] = run_path.with_name(replacement_name)

        if create_path and not run_path.is_dir():
            run_path.mkdir(parents=True)
        return run_path

    def track_run(
            self,
            registration_key: Registration,
            serialization_path: Path
    ):
        """
        Tracks a run by storing its metadata in a file.

        Args:
            registration_key: configuration registration key
            serialization_path: path where run results are stored
        """

        data = {
            'name': registration_key.name,
            'tags': registration_key.tags,
            'namespace': registration_key.namespace,
            'run_directory': serialization_path.name,
            'date': datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        }
        tracker_path: Path = self.runs_directory / self.run_tracker_filename
        if not tracker_path.exists():
            df = pd.DataFrame(data=[data])
        else:
            df = pd.read_csv(tracker_path)
            df = df.append([data])

        df.to_csv(tracker_path, index=False)

    def run(
            self,
            filepath: Optional[Union[AnyStr, Path]] = None,
            strict_existence: bool = False,
            create_path: bool = True
    ) -> Path:
        """
        The ``FileManager`` receives a filepath and updates it according to its initialized ``base_directory``
        (see ``setup()``).

        Args:
            filepath: path to file that has to be updated according to ``base_directory``
            strict_existence: if True, the ``FileManager`` expects that the updated filepath already exists.
            create_path: if True, the ``FileManager`` checks if built path has to be created

        Returns:
             Updated ``filepath`` relative to ``base_directory``

        Raises:
            ``FileNotFoundError``: if ``strict_existence = True`` and the update filepath does not exist.
        """
        if self.base_directory is None:
            logging_utility.logger.warning(f"base_directory is None! Setting base_directory to current directory...")
            self.base_directory = Path('')

        built_path: Path = self.base_directory.joinpath(filepath)
        path_exists = built_path.is_dir()
        if strict_existence and not path_exists:
            raise FileNotFoundError(f'Expected path={built_path} to exist and it is not.')

        if create_path and not path_exists:
            built_path.mkdir()

        return built_path
