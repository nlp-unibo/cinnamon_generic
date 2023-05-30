import abc
from pathlib import Path
from typing import AnyStr, Any, Optional, Union

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from dotmap import DotMap

from cinnamon_generic.components.callback import Callback, guard
from cinnamon_generic.components.metrics import Metric


# TODO: check whether to split training/inference APIs into a Trainer/Predictor dedicated component.
class Model(Component):
    """
    A ``Model`` is a ``Component`` specialized for wrapping a machine learning model.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model: Any = None

    # General

    @abc.abstractmethod
    def save_model(
            self,
            filepath: Union[AnyStr, Path]
    ):
        """
        Serializes internal model's state to filesystem.

        Args:
            filepath: path where to save model's state.
        """
        pass

    @abc.abstractmethod
    def load_model(
            self,
            filepath: Union[AnyStr, Path]
    ):
        """
        Loads internal model's state from a serialized checkpoint stored in the filesystem.

        Args:
            filepath: path where the model serialized checkpoint is stored.
        """

        pass

    def save(
            self,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
            overwrite: bool = False
    ):
        """
        Serializes the ``Model``'s state to filesystem.

        Args:
            serialization_path: Path where to save the ``Model`` state and its internal model.
            overwrite: if True, the existing serialized ``Model`` data is overwritten.
        """

        super().save(serialization_path=serialization_path)
        self.save_model(filepath=serialization_path)

    def load(
            self,
            serialization_path: Optional[Union[AnyStr, Path]] = None
    ):
        """
        Loads the ``Model``'s state from a serialized checkpoint stored in the filesystem.

        Args:
            serialization_path: path where the model serialized checkpoint is stored.
        """

        super().load(serialization_path=serialization_path)
        self.model = self.load_model(filepath=serialization_path)

    @property
    def state(
            self
    ) -> DotMap:
        return_dict = {key: value for key, value in self.__dict__.items() if key != 'model'}
        return DotMap(return_dict)

    def prepare_for_training(
            self,
            train_data: FieldDict
    ):
        """
        Entry point for preparing the model before training.
        For instance, computing class weights for regularizing an imbalanced classification setting
        can be computed here.

        Args:
            train_data: a ``FieldDict`` representing training data. Each key is a specific input.
        """

        pass

    def prepare_for_loading(
            self,
            data: Optional[FieldDict] = None
    ):
        """
        Entry point for preparing the model before loading the internal model's state.

        Args:
            data: a ``FieldDict`` representing data that can be used to set up the model.
        """

        pass

    def check_after_loading(
            self
    ):
        """
        Entry point for validating the model after state loading.
        """

        pass

    @abc.abstractmethod
    def build_model(
            self,
            processor_state: FieldDict,
            callbacks: Optional[Callback] = None
    ):
        """
        Builds the internal model.

        Args:
            processor_state: a ``FieldDict`` storing all previous components' relevant information for
             building the model.
            callbacks: an optional ``Callback`` component.
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_fit')
    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        """
        Fits the model with given training and (optionally) validation data.

        Args:
            train_data: training data necessary for training the model
            val_data: validation data that can be used to regularize or monitor the training process
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process

        Returns:
            A ``FieldDict`` storing training information
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_evaluate')
    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        """
        Evaluates a trained model on given data.

        Args:
            data: data to evaluate the model on
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process

        Returns:
            A ``FieldDict`` storing evaluation information
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_evaluate_and_predict')
    def evaluate_and_predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        """
        Evaluates a trained model on given data and computes model predictions on the same data.

        Args:
            data: data to evaluate the model on and compute predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process

        Returns:
            A ``FieldDict`` storing evaluation and prediction information
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_predict')
    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        """
        Computes model predictions on the given data.

        Args:
            data: data to compute model predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process

        Returns:
            A ``FieldDict`` storing prediction information
        """

        pass

    @abc.abstractmethod
    def get_model_data(
            self,
            data: FieldDict,
            with_labels: bool = False
    ) -> Any:
        """
        Generic entrypoint to define data iterators depending on the internal model APIs.

        Args:
            data: data to feed to the model
            with_labels: if True, ground-truth information is considered (i.e., the model is in training mode)

        Returns:
            Data formatted to be compliant with the internal model APIs.
        """

        pass

    def run(
            self,
            data: Optional[FieldDict] = None
    ) -> FieldDict:
        """
        The default ``Model`` behaviour is to execute ``evaluate_and_predict`` given input data.

        Args:
            data: data to evaluate the model on and compute predictions.

        Returns:
            A ``FieldDict`` storing evaluation and prediction information
        """

        return self.evaluate_and_predict(data=data)


class Network(Model):
    """
    A ``Network`` is a ``Model`` extension specialized for wrapping neural networks.
    """

    @abc.abstractmethod
    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[FieldDict] = None
    ) -> Any:
        """
        Computes the training loss of the neural network

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            The computed training loss for the current step
        """

        pass

    @abc.abstractmethod
    def batch_train(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[FieldDict] = None
    ) -> Any:
        """
        Computes a training step given input data.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            The computed training information for the current step (e.g., loss value and gradients)
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_batch_fit')
    def batch_fit(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[FieldDict] = None
    ) -> Any:
        """
        Computes a batch fitting step given input batch

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            The computed training information for the current step (e.g., loss value, metadata, gradients)
        """
        pass

    @abc.abstractmethod
    @guard(hookpoint='on_batch_predict')
    def batch_predict(
            self,
            batch_x: Any,
            batch_args: Optional[FieldDict] = None
    ):
        """
        Computes model predictions for the given input batch.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            Model predictions for the given input batch
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_batch_evaluate')
    def batch_evaluate(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[FieldDict] = None
    ):
        """
        Computes training loss for the given input batch without a issuing a gradient step.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            Training loss for the given input batch
        """

        pass

    @abc.abstractmethod
    @guard(hookpoint='on_batch_evaluate_and_predict')
    def batch_evaluate_and_predict(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[FieldDict] = None
    ):
        """
        Computes training loss and model predictions for the given input batch without a issuing a gradient step.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            batch_args: additional information for the specific step.

        Returns:
            Training loss and model predictions for the given input batch
        """

        pass
