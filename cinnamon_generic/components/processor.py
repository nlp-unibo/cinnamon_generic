from typing import Optional

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict


# TODO: add possibility to run processor in-place or not (overwrite or not data)
class Processor(Component):
    """
    A ``Processor`` is a ``Component`` that is specialized in processing input data.
    It is mainly used to prepare input data for a model after having loaded it via a ``DataLoader`` component.
    """

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        """
        Processes input data.

        Args:
            data: data to process in ``FieldDict`` format.
            is_training_data: if True, input data comes from the training split.

        Returns:
            The processed version of input ``data``
        """
        return data

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        """
        The default behaviour of ``Processor`` is to run ``process()`` for given input data.

        Args:
            data: data to process in ``FieldDict`` format.
            is_training_data: if True, input data comes from the training split.

        Returns:
            The processed version of input ``data`` if it is not None.
        """

        if data is None:
            return data

        data = self.process(data=data, is_training_data=is_training_data)

        return data


class ProcessorPipeline(Processor):
    """
    A pipeline of ``Processor`` components to be executed in a sequential fashion.
    """

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        processor = self.config.search_by_tag()

        for processor in self.processors:
            data = processor.run(data=data,
                                 is_training_data=is_training_data)
        return data
