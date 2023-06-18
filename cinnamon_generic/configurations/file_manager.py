from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register

from cinnamon_generic.components.file_manager import FileManager


class FileManagerConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        # Directories
        config.add(name='logging_directory',
                   value='logging',
                   type_hint=str,
                   tags={'directory'},
                   description="directory name for library logger")
        config.add(name='dataset_directory',
                   type_hint=str,
                   value='datasets',
                   tags={'directory'},
                   description="directory name for storing datasets")
        config.add(name='registrations_directory',
                   type_hint=str,
                   value='registrations',
                   tags={'directory'},
                   description="directory name for storing components and configurations registrations")
        config.add(name='calibrations_directory',
                   type_hint=str,
                   value='calibrations',
                   tags={'directory'},
                   description="directory name for storing calibrated configurations")
        config.add(name='calibration_runs_directory',
                   type_hint=str,
                   value='calibration_runs',
                   tags={'directory'},
                   description="directory name for storing calibration tasks")
        config.add(name='routine_data_directory',
                   type_hint=str,
                   value='routine_data',
                   tags={'directory'},
                   description="directory name where pre-computed routine data "
                               "(e.g., pre-built cv folds) is stored")
        config.add(name='runs_directory',
                   type_hint=str,
                   value='runs',
                   tags={'directory'},
                   description="directory name where Component results are stored")

        # Filenames
        config.add(name='logging_filename',
                   type_hint=str,
                   value='daily_log.log',
                   tags={'filename'},
                   description="filename for logging")
        config.add(name='calibration_results_filename',
                   type_hint=str,
                   tags={'filename'},
                   value='calibration_results.json',
                   description="filename of summary file storing calibrated model configurations.")
        config.add(name='run_tracker_filename',
                   type_hint=str,
                   tags={'filename'},
                   value='runs_info.csv',
                   description="filename of summary file storing mapping between "
                               "run folders and registration keys.")

        return config


@register
def register_file_managers():
    Registry.add_and_bind(config_class=FileManagerConfig,
                          component_class=FileManager,
                          name='file_manager',
                          namespace='generic',
                          is_default=True)
