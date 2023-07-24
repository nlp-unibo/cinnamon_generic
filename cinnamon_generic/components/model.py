import abc
from pathlib import Path
from typing import AnyStr, Any, Optional, Union, Dict

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.callback import Callback, guard
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.processor import Processor


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
            name: Optional[str] = None
    ):
        """
        Serializes the ``Model``'s state to filesystem.

        Args:
            serialization_path: Path where to save the ``Model`` state and its internal model.
            name: if the component is a child in another component configuration, ``name`` is the parameter key
            used by the parent to reference the child. Otherwise, ``name`` is automatically set to the component class
            name.
        """

        super().save(serialization_path=serialization_path)
        self.save_model(filepath=serialization_path)

    def load(
            self,
            serialization_path: Optional[Union[AnyStr, Path]] = None,
            name: Optional[str] = None
    ):
        """
        Loads the ``Model``'s state from a serialized checkpoint stored in the filesystem.

        Args:
            serialization_path: path where the model serialized checkpoint is stored.
            name: if the component is a child in another component configuration, ``name`` is the parameter key
            used by the parent to reference the child. Otherwise, ``name`` is automatically set to the component class
            name.
        """

        super().load(serialization_path=serialization_path)
        self.model = self.load_model(filepath=serialization_path)

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
    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        """
        Builds the internal model.

        Args:
            processor: a ``FieldDict`` storing all previous components' relevant information for
             building the model.
            callbacks: an optional ``Callback`` component.
        """

        pass

    @abc.abstractmethod
    @guard()
    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        """
        Fits the model with given training and (optionally) validation data.

        Args:
            train_data: training data necessary for training the model
            val_data: validation data that can be used to regularize or monitor the training process
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.

        Returns:
            A ``FieldDict`` storing training information
        """
        pass

    @abc.abstractmethod
    @guard()
    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        """
        Evaluates a trained model on given data and computes model predictions on the same data.

        Args:
            data: data to evaluate the model on and compute predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.
            suffixes: suffixes used to uniquely identify evaluation results on input data

        Returns:
            A ``FieldDict`` storing evaluation and prediction information
        """

        pass

    @abc.abstractmethod
    @guard()
    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        """
        Computes model predictions on the given data.

        Args:
            data: data to compute model predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.
            suffixes: suffixes used to uniquely identify evaluation results on input data

        Returns:
            A ``FieldDict`` storing prediction information
        """

        pass

    def run(
            self,
            data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        """
        The default ``Model`` behaviour is to execute ``evaluate_and_predict`` given input data.

        Args:
            data: data to evaluate the model on and compute predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.

        Returns:
            A ``FieldDict`` storing evaluation and prediction information
        """

        return self.evaluate(data=data)


class Network(Model):
    """
    A ``Network`` is a ``Model`` extension specialized for wrapping neural networks.
    """

    @abc.abstractmethod
    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
    ) -> Any:
        """
        Computes the training loss of the neural network

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format

        Returns:
            The computed training loss for the current step
        """

        pass

    @abc.abstractmethod
    def batch_train(
            self,
            batch_x: Any,
            batch_y: Any,
    ) -> Any:
        """
        Computes a training step given input data.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format

        Returns:
            The computed training information for the current step (e.g., loss value and gradients)
        """

        pass

    @abc.abstractmethod
    def batch_fit(
            self,
            batch_x: Any,
            batch_y: Any,
    ) -> Any:
        """
        Computes a batch fitting step given input batch

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format

        Returns:
            The computed training information for the current step (e.g., loss value, metadata, gradients)
        """
        pass

    @abc.abstractmethod
    def batch_predict(
            self,
            batch_x: Any,
    ):
        """
        Computes model predictions for the given input batch.

        Args:
            batch_x: batch input training data in any model-compliant format

        Returns:
            Model predictions for the given input batch
        """

        pass

    @abc.abstractmethod
    def batch_evaluate(
            self,
            batch_x: Any,
            batch_y: Any,
    ):
        """
        Computes training loss for the given input batch without a issuing a gradient step.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format

        Returns:
            Training loss for the given input batch
        """

        pass
