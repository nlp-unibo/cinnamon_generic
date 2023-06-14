import os
import shutil
from pathlib import Path
from typing import AnyStr, List, Union, Optional, Callable, Dict, Any, Tuple

from tqdm import tqdm

from cinnamon_core.core.data import FieldDict, ValidationFailureException
from cinnamon_core.core.registry import RegistrationKey, Registry, Registration, Tag, InvalidConfigurationTypeException
from cinnamon_core.utility import logging_utility
from cinnamon_core.utility.json_utility import load_json, save_json
from cinnamon_core.utility.python_utility import get_function_signature
from cinnamon_generic.components.file_manager import FileManager


def retrieve_and_save(
        save_folder: Path,
        conditions: List[Callable[[RegistrationKey], bool]]
):
    registration_keys = sorted([str(key) for key in Registry.REGISTRY if all([cond(key) for cond in conditions])])

    valid_keys = []
    invalid_keys = []
    for key in registration_keys:
        try:
            Registry.build_component_from_key(config_registration_key=key)
            valid_keys.append(key)
        except (InvalidConfigurationTypeException, ValidationFailureException):
            invalid_keys.append(key)

    if valid_keys:
        save_json(save_folder.joinpath('valid.json'), valid_keys)

    if invalid_keys:
        save_json(save_folder.joinpath('invalid.json'), invalid_keys)


def find_modules(
        root: Union[Path, AnyStr]
) -> List[Path]:
    root = Path(root) if type(root) != Path else root
    folders = [item for item in root.rglob(pattern='**/') if item.is_dir() and item.name.casefold() != '__pycache__']
    # Ignore first item (root)
    return folders[1:]


# Commands

def setup_registry(
        directory: Union[Path, AnyStr] = None,
        module_directories: List[Union[AnyStr, Path]] = None,
        registrations_to_file: bool = False,
        file_manager_registration_key: Registration = None
) -> Union[RegistrationKey, str]:
    """
    This command does the following actions:
    - Populates the ``Registry`` with specified registration actions.
    - Builds the ``FileManager`` ``Component`` and stores its instance in the ``Registry`` for quick use.
    - Set-ups the logging utility module.
    - If ``generate_registration``, invokes the ``list_registrations`` command for debugging and readability purposes.

    !IMPORTANT!: this command is always required at beginning of each of your scripts for proper
    ``Registry`` initialization.

    Args:
        directory: path to the base directory where to look for standard ``FileManager`` folders.
        module_directories: list of base directories where to look for registration calls.
        registrations_to_file: if True, the ``list_registrations`` command is invoked.
        file_manager_registration_key:

    Returns:
        The built ``FileManager``'s ``RegistrationKey``
    """

    directory = Path(directory).resolve() if type(directory) != Path else directory
    Registry.load_registrations(directory_path=directory)

    if module_directories is not None:
        for mod_dir in module_directories:
            mod_dir = Path(mod_dir).resolve() if type(directory) != Path else mod_dir
            for module_name in find_modules(root=mod_dir):
                Registry.load_registrations(directory_path=module_name)

    Registry.check_registration_graph()
    Registry.expand_and_resolve_registration()

    if file_manager_registration_key is None:
        file_manager_registration_key = RegistrationKey(name='file_manager',
                                                        tags={'default'},
                                                        namespace='generic')

    file_manager = FileManager.build_component_from_key(config_registration_key=file_manager_registration_key,
                                                        register_built_component=True)
    file_manager.setup(base_directory=directory)

    logging_path = file_manager.run(filepath=file_manager.logging_directory)
    logging_path = logging_path.joinpath(file_manager.logging_filename)
    logging_utility.set_logging_path(logging_path=logging_path)
    logging_utility.build_logger(__name__)

    if registrations_to_file:
        serialize_registrations()

    return file_manager_registration_key


def serialize_registrations(
        namespaces: Optional[List[str]] = None
):
    """
    Retrieves all registered ``Configuration`` in the ``Registry`` and serializes the corresponding ``RegistrationKey``
    to file. ``RegistrationKey`` are organized by namespace.

    Args:
        namespaces: if provided, only the registrations under specified namespaces are serialized
        (useful for debugging purposes).
    """
    file_manager = FileManager.retrieve_built_component(name='file_manager',
                                                        namespace='generic',
                                                        is_default=True)

    logging_utility.logger.info(f'Saving registration info to folder: {file_manager.registrations_directory}')

    if namespaces is None:
        logging_utility.logger.info('No namespace set specified. Retrieving all available namespaces...')
        namespaces = set([key.namespace for key in Registry.REGISTRY])

    logging_utility.logger.info(f'Total namespaces: {len(namespaces)}{os.linesep}'
                                f'Namespaces: {os.linesep}'
                                f'{namespaces}')
    registration_directory = file_manager.run(filepath=file_manager.registrations_directory)

    if registration_directory.exists():
        shutil.rmtree(registration_directory)

    for namespace in tqdm(namespaces):
        namespace_registration_path = registration_directory.joinpath(namespace)

        if not namespace_registration_path.is_dir():
            namespace_registration_path.mkdir(parents=True)

        retrieve_and_save(save_folder=namespace_registration_path,
                          conditions=[lambda key: key.namespace == namespace])


def run_component_from_key(
        config_registration_key: Registration,
        serialize: bool = False,
        run_name: Optional[str] = None,
        run_args: Optional[Dict] = None
) -> Tuple[Any, Optional[Path]]:
    logging_utility.logger.info(f'Retrieving Component from key:{os.linesep}{config_registration_key}')

    component = Registry.build_component_from_key(config_registration_key=config_registration_key)

    file_manager = FileManager.retrieve_built_component(name='file_manager',
                                                        namespace='generic',
                                                        is_default=True)

    if serialize:
        serialization_path = file_manager.register_temporary_run_name(replacement_name=run_name,
                                                                      create_path=serialize,
                                                                      key=config_registration_key)
        logging_utility.update_logger(serialization_path.joinpath(file_manager.logging_filename))
    else:
        serialization_path = None

    run_args = run_args if run_args is not None else {}
    if 'serialization_path' in get_function_signature(component.run):
        run_args['serialization_path'] = serialization_path
    run_result = component.run(**run_args)

    if serialize:
        logging_utility.logger.info(f'Serializing Component state to: {serialization_path}')
        component.save(serialization_path=serialization_path)

    if serialization_path is None:
        return run_result

    if run_name is not None and serialization_path.exists():
        replacement_path: Path = file_manager.runs_registry[serialization_path]
        if replacement_path.exists():
            logging_utility.logger.warning(
                f'Replacement path {replacement_path} already exists! Skipping replacement...')
        else:
            serialization_path.rename(replacement_path)
            serialization_path = replacement_path
            logging_utility.logger.info(f'Renaming {serialization_path} to {replacement_path}')

    if serialization_path.exists():
        save_json(serialization_path.joinpath('metadata.json'),
                  data={
                      'config_registration_key': str(config_registration_key),
                      'config': component.config.to_value_dict()
                  },
                  unpicklable=False)
        file_manager.track_run(registration_key=config_registration_key,
                               serialization_path=serialization_path)

    return run_result, serialization_path


def run_component(
        name: str,
        tags: Tag = None,
        namespace: str = 'generic',
        serialize: bool = False,
        run_name: Optional[str] = None,
        run_args: Optional[Dict] = None
) -> Tuple[Any, Optional[Path]]:
    key = RegistrationKey(name=name,
                          tags=tags,
                          namespace=namespace)
    return run_component_from_key(config_registration_key=key,
                                  serialize=serialize,
                                  run_name=run_name,
                                  run_args=run_args)


def routine_train(
        name: str,
        tags: Tag = None,
        namespace: str = 'generic',
        serialize: bool = False,
        run_name: Optional[str] = None,
        run_args: Optional[Dict] = None
) -> FieldDict:
    """
    Builds a ``Routine`` ``Component`` and runs it in training mode.

    Args:
        name:
        tags:
        namespace:
        serialize: if True, it enables the serialization process of ``Routine`` component during execution.
        run_name:
        run_args:
    """
    routine_key = RegistrationKey(name=name, tags=tags, namespace=namespace)
    return routine_train_from_key(routine_registration_key=routine_key,
                                  serialize=serialize,
                                  run_name=run_name,
                                  run_args=run_args)


def routine_train_from_key(
        routine_registration_key: Registration,
        serialize: bool = False,
        run_name: Optional[str] = None,
        run_args: Optional[Dict] = None
) -> FieldDict:
    routine_args = {
        'is_training': True
    }
    run_args = {**routine_args, **run_args} if run_args is not None else routine_args
    routine_result, serialization_path = run_component_from_key(config_registration_key=routine_registration_key,
                                                                run_name=run_name,
                                                                serialize=serialize,
                                                                run_args=run_args)

    if serialization_path is not None:
        save_json(serialization_path.joinpath('result.json'), routine_result.to_value_dict())

    return routine_result


def routine_multiple_train(
        routine_registration_keys: List[Registration],
        serialize: bool = False
) -> List[FieldDict]:
    """
    Sequentially executes the ``train`` command for each specified ``Routine`` ``RegistrationKey``.

    Args:
        routine_registration_keys: a list of ``Routine`` ``RegistrationKey`` instances
        serialize: if True, it enables the serialization process of ``Routine`` component during execution.
    """

    result = []
    for routine_registration_key in routine_registration_keys:
        run_result = routine_train_from_key(routine_registration_key=routine_registration_key,
                                            serialize=serialize)
        result.append(run_result)
    return result


def routine_inference(
        routine_path: Optional[Union[AnyStr, Path]] = None,
        run_name: Optional[str] = None,
        namespace: Optional[str] = None,
        serialize: bool = False
) -> FieldDict:
    """
    Builds a ``Routine`` ``Component`` and runs it in inference mode.

    Args:
        routine_path: path where ``Routine`` training result is stored
        run_name: directory name under 'pipelines' folder where ``Routine`` training result is stored.
        serialize: if True, it enables the serialization process of ``Routine`` component during execution.
        namespace: TODO

    Raises
        ``AttributeError``: if both ``routine_path`` and ``routine_name`` are not specified.
        ``FileNotFoundError``: if no ``train`` command metadata file is found.
    """
    if routine_path is None and run_name is None:
        raise AttributeError('At least routine_path or run_name have to be specified.'
                             f'Got routine_path={routine_path} and run_name={run_name}')

    file_manager = FileManager.retrieve_built_component(name='file_manager',
                                                        namespace='generic',
                                                        is_default=True)

    if routine_path is None:
        routine_path = file_manager.run(filepath=file_manager.routine_directory)
        routine_path = routine_path.joinpath(namespace, 'routine', run_name)

    metadata_path = routine_path.joinpath('metadata.json')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Expected to find metadata file {metadata_path}...')

    command_metadata_info = load_json(metadata_path)
    routine_registration_key = RegistrationKey.from_string(command_metadata_info['routine_registration_key'])

    # Sanity check
    assert routine_registration_key.namespace == namespace, \
        f'Found inconsistent namespaces. Given {namespace} != Found {routine_registration_key.namespace}'

    routine_result, serialization_path = run_component_from_key(config_registration_key=routine_registration_key,
                                                                serialize=serialize,
                                                                run_name=run_name,
                                                                run_args={
                                                                    'is_training': False
                                                                })
    return routine_result


def routine_multiple_inference(
        routine_paths: Optional[List[Union[AnyStr, Path]]] = None,
        routine_names: Optional[List[str]] = None,
        serialize: bool = False
) -> List[FieldDict]:
    """
    Sequentially executes the ``inference`` command for each specified ``Routine`` ``RegistrationKey``.

    Args:
        routine_paths: list of paths where ``Routine`` training result is stored
        routine_names: list of directory names under 'pipelines' folder where ``Routine`` training result is stored.
        serialize: if True, it enables the serialization process of ``Routine`` component during execution.

    Raises
        ``AttributeError``: if both ``routine_paths`` and ``routine_names`` are not specified.
    """

    if routine_paths is None and routine_names is None:
        raise AttributeError('At least routine_paths or routine_names have to be specified.'
                             f'Got routine_paths={routine_paths} and routine_names={routine_names}')

    if routine_paths is not None:
        routine_names = [None] * len(routine_paths)
    else:
        routine_paths = [None] * len(routine_names)

    result = []
    for routine_path, routine_name in zip(routine_paths, routine_names):
        run_result = routine_inference(routine_path=routine_path,
                                       run_name=routine_name,
                                       serialize=serialize)
        result.append(run_result)
    return result


__all__ = [
    'setup_registry',
    'serialize_registrations',
    'run_component',
    'run_component_from_key',
    'routine_train',
    'routine_multiple_train',
    'routine_inference',
    'routine_multiple_inference'
]
