.. _data_loader:

Data Loader
*************************************

As the name says, a ``DataLoader`` loads some data.

In particular, a ``DataLoader`` does the following:

- Loads data to build a proper dataset
- Defines train, validation and test splits (if available)
- Parses each data split to define the desired set of input data to pass to other components.

In other terms, a ``DataLoader`` loads some data and returns a pre-processed view of such data that is compliant with other components' requirements.

The following APIs are provided:

.. code-block:: python

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

        # Get splits
        train_data, val_data, test_data = self.get_splits()

        if not self.has_test_split_only:
            if train_data is None:
                raise UnspecifiedDataSplitException(split='training')

            if self.has_val_split and val_data is None:
                raise UnspecifiedDataSplitException(split='validation')

        if self.has_test_split and test_data is None:
            raise UnspecifiedDataSplitException(split='test')

        # Provide a view for each split via FieldDict
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

The ``DataLoader`` uses the ``DataLoaderConfig`` as default configuration template:

.. code-block:: python

    class DataLoaderConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='name',
                       type_hint=str,
                       description="Unique dataset identifier",
                       is_required=True)

            config.add(name='has_test_split_only',
                       value=False,
                       type_hint=bool,
                       description="Whether the ``DataLoader`` has test split only or not")

            config.add(name='has_val_split',
                       value=True,
                       type_hint=bool,
                       description="Whether the ``DataLoader`` has a val split or not")

            config.add(name='has_test_split',
                       value=True,
                       type_hint=bool,
                       description="Whether the ``DataLoader`` has a test split or not")

            return config