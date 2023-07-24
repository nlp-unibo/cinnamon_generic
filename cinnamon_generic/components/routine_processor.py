import abc
from typing import Dict

import pandas as pd

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.processor import Processor


class RoutineProcessor(Processor):

    @abc.abstractmethod
    def accumulate(
            self,
            accumulator: Dict,
            step: FieldDict
    ) -> Dict:
        """
        Accumulates processed information into ``accumulator`` given ``step`` input data.

        Args:
            accumulator: a dictionary containing accumulated processed data
            step: routine step information regarding a fold.

        Returns:
            Accumulated processed information
        """

        pass

    @abc.abstractmethod
    def aggregate(
            self,
            info: Dict
    ) -> Dict:
        """
        Aggregates accumulated information for visualization and summary purposes.

        Args:
            info: accumulated processed information

        Returns:
            Aggregated processed information
        """
        pass


class AverageProcessor(RoutineProcessor):
    """
    A ``RoutineProcessor`` that computes average and std for each loss and metric.
    """

    def accumulate(
            self,
            accumulator: Dict,
            step: FieldDict
    ) -> Dict:
        """
        Accumulates loss and metric information over fold steps.
        In particular, the following accumulation data structure is defined:
        _____________________________________________________________________________
        | metric_name | metric_value | info_key | suffix1 | suffix2 | ... | suffixN |
        |   ...             ...          ...       ...       ...      ...     ...   |
        |                                                                           |
        |___________________________________________________________________________|

        Args:
            accumulator: a dictionary containing accumulated processed data
            step: routine step information regarding a fold.

        Returns:
            Accumulated processed information
        """

        routine_suffixes = step.search_by_tag(tags={'routine_suffix'},
                                              exact_match=True)
        for info_key, info in step.search_by_tag(tags={'info'},
                                                 exact_match=True).items():

            for key, value in info.to_value_dict().items():
                if type(value) == float:
                    accumulator.setdefault('metric_name', []).append(key)
                    accumulator.setdefault('metric_value', []).append(value)
                    accumulator.setdefault('info_key', []).append(info_key)

                    for suffix_name, suffix_value in routine_suffixes.items():
                        accumulator.setdefault(f'suffix_{suffix_name}', []).append(suffix_value)

                if key == 'metrics':
                    for metric_name, metric_value in info.metrics.items():
                        accumulator.setdefault('metric_name', []).append(metric_name)
                        accumulator.setdefault('metric_value', []).append(metric_value)
                        accumulator.setdefault('info_key', []).append(info_key)

                        for suffix_name, suffix_value in routine_suffixes.items():
                            accumulator.setdefault(f'suffix_{suffix_name}', []).append(suffix_value)

        return accumulator

    def aggregate(
            self,
            info: Dict
    ) -> Dict:
        """
        Aggregates loss and metric information by computing the average and std over steps.

        Args:
            info: accumulated processed information

        Returns:
            The average and std for each loss and metric
        """

        df_view = pd.DataFrame.from_dict(info)
        df_view = df_view.groupby(['metric_name', 'info_key'])
        average = df_view['metric_value'].mean()
        average.name = 'average'
        std = df_view['metric_value'].std()
        std.name = 'std'

        merged = pd.merge(average, std, left_index=True, right_index=True)
        return merged.to_dict()

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        """
        Processes ``Routine`` result to compute average and std for each loss and metric.

        Args:
            data: ``Routine`` result
            is_training_data: if True, input data comes from the training split.

        Returns:

        """

        average_data = {}
        for step in data.steps:
            average_data = self.accumulate(accumulator=average_data,
                                           step=step)

        average_data = self.aggregate(info=average_data)
        data.add(name='average',
                 value=average_data,
                 description=f'Processed routine results via {self.__class__.__name__}.'
                             f'Each metric is averaged across routine suffixes.')
        return data


class FoldProcessor(AverageProcessor):
    """
    A ``AverageProcessor`` that computes average and std for each loss and metric over cross-validation folds.
    """

    def aggregate(
            self,
            info: Dict
    ) -> Dict:
        """
        Aggregates loss and metric information by computing the average and std over fold steps and suffixes.
        In addition, it computes average and std over each suffix individually (e.g., seeds, fold).

        Args:
            info: accumulated processed information

        Returns:
            The average and std for each loss and metric
        """

        df_view = pd.DataFrame.from_dict(info)
        routine_suffixes = [col for col in df_view if col.startswith('suffix_')]

        aggregate_data = {}
        for routine_suffix in routine_suffixes:
            suffix_view = df_view.groupby(['metric_name', 'info_key', routine_suffix])
            average = suffix_view['metric_value'].mean()
            average.name = 'average'
            std = suffix_view['metric_value'].std()
            std.name = 'std'

            merged = pd.merge(average, std, left_index=True, right_index=True)
            aggregate_data.setdefault(routine_suffix, merged.to_dict())

        aggregate_data.setdefault('all', super().aggregate(info))

        return aggregate_data
