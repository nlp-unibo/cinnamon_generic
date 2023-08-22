.. _model:

Model
*************************************

A ``Model`` is a ``Component`` specialized for wrapping a machine learning model.

The ``Model`` component defines the following APIs:

- ``save`` and ``load``: for saving/loading the component.
- ``save_model`` and ``load_model``: for saving/loading the internal model.
- ``prepare_for_training``: entrypoint for preparing the model before training (e.g., computing class weights).
- ``prepare_for_loading``: entrypoint for preparing the model before loading the internal model's state.
- ``build``: builds the internal model.
- ``fit``: fits the model with given training and (optionally) validation data.
- ``evaluate``: evaluates a trained model on given data and computes model predictions on the same data.
- ``predict``: computes model predictions on the given data.

*************************************
Network
*************************************

A ``Network`` is a ``Model`` extension specialized for wrapping neural networks.

The ``Network`` component defines the following APIs:

- ``batch_loss``: computes the training loss of the neural network.
- ``batch_train``: computes a training step given input data.
- ``batch_fit``: computes a batch fitting step given input batch.
- ``batch_evaluate``: computes training loss for the given input batch without a issuing a gradient step.
- ``batch_predict``: computes model predictions for the given input batch.

The ``Network`` uses ``NetworkConfig`` as the default configuration template:

.. code-block:: python

    class NetworkConfig(TunableConfiguration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='epochs',
                       is_required=True,
                       description='Number of epochs to perform to train a model',
                       type_hint=int,
                       allowed_range=lambda epochs: epochs > 0)

            config.add(name='stop_training',
                       value=False,
                       description='If enabled, the model stops training immediately',
                       type_hint=bool)

            return config