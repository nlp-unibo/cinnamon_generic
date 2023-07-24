from typing import Optional

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.pipeline import OrderedPipeline


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

    def finalize(
            self
    ):
        """
        Finalizes the ``Processor`` internal state.
        This functions should be called when no other run() function calls are required.
        """

        pass

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


class ProcessorPipeline(OrderedPipeline, Processor):

    def finalize(
            self
    ):
        components = self.get_pipeline()
        for component in components:
            component.finalize()

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> FieldDict:
        components = self.get_pipeline()
        for component in components:
            data = component.run(data=data, is_training_data=is_training_data)
        return data
