import abc
from typing import Tuple, Any, Optional

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict


class UnspecifiedDataSplitException(Exception):

    def __init__(self, split: str):
        super().__init__(f'At least a {split} set should be provided')


class DataLoader(Component):
    """
    Generic ``Component`` for data loading.
    The ``DataLoader`` does the following:
     - Loads data to build a proper dataset
     - Defines train, validation and test splits
     - Parses each data split to define the desired set of input data to pass to other components.
    """

    @abc.abstractmethod
    def load_data(
            self
    ) -> Any:
        """
        Loads data to build a proper dataset to work on.

        Returns:
            The built dataset in any desired format.
        """
        pass

    @abc.abstractmethod
    def get_splits(
            self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """
        Returns the train, validation and test splits of the loaded dataset.
        Depending on its usage, any combination of data splits may be optional (i.e., None)

        Returns:
            A tuple of train, validation and test splits.
        """
        pass

    @abc.abstractmethod
    def parse(
            self,
            data: Optional[Any] = None,
    ) -> Optional[FieldDict]:
        """
        Parses a data split to define the desired inputs

        Args:
            data: input data split to parse

        Returns:
            A ``FieldDict`` instance containing the desired inputs.
        """
        pass

    def run(
            self
    ) -> FieldDict:
        """
        Defines and parses the train, validation and test splits of the loaded dataset.

        Returns:
            A ``FieldDict`` instance containing the parsed data splits, accessible via
            ``.train``, ``.val`` and ``.test`` field names.

            For instance:
            ``
            data_loader = DataLoader()
            data = data_loader.run()
            train_data = data.train
            val_data = data.val
            test_data = data.test
            ``

            Note that each of the above data splits can be None

        Raises:
            ``UnspecifiedDataSplitException``: if a data split is expected, but it is None
        """

        train_data, val_data, test_data = self.get_splits()

        # We might use a DataLoader to load inference data only
        if not self.has_test_split_only:
            if train_data is None:
                raise UnspecifiedDataSplitException(split='training')

            if self.has_val_split and val_data is None:
                raise UnspecifiedDataSplitException(split='validation')

        if self.has_test_split and test_data is None:
            raise UnspecifiedDataSplitException(split='test')

        # Build instances
        result = FieldDict()
        if train_data is not None:
            train_data = self.parse(data=train_data)
        result.add(name='train',
                   value=train_data,
                   type_hint=FieldDict,
                   tags={'train'})
        if val_data is not None:
            val_data = self.parse(data=val_data)
        result.add(name='val',
                   value=val_data,
                   type_hint=FieldDict,
                   tags={'val'})
        if test_data is not None:
            test_data = self.parse(data=test_data)
        result.add(name='test',
                   value=test_data,
                   type_hint=FieldDict,
                   tags={'test'})

        return result
