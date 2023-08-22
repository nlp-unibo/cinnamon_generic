.. _callback:

Callbacks
*************************************

A ``Callback`` component defines execution flow hookpoints for flow customization and side effects.

Callbacks are designed to behave just like any `Keras <https://keras.io/>`_ callback.

In particular, users are **free to define** callback hookpoints.

In cinnamon, hookpoints can be used in two different ways:

- guard hookpoint
- manual hookpoint

--------------------------------------
Guard hookpoint
--------------------------------------

a guard hookpoint is a pair of function with the following prefixes:

.. code-block:: python

    def on_hookpoint_begin()
        ...

    def on_hookpoint_end()
        ...

The hookpoint pair wraps a function to perform actions at its beginning and end.

To quickly use the above hookpoint pair, we can decorate our wrapped function with ``@guard`` decorator.

.. note::
    The wrapped functions needs to have ``callbacks`` argument in its signature.
    In particular, ``callbacks`` must be of type ``Callback``.

.. code-block:: python

    @guard('on_hookpoint')
    def my_function(*args, **kwargs, callbacks):
        ...

When we invoke ``my_function``, the ``callbacks`` argument is retrieved from its signature and the specified guard hookpoint is issued at the beginning and end of ``my_function``.


--------------------------------------
Manual hookpoint
--------------------------------------

Manual hookpoint are simply explicit hookpoint method call in user script.

.. code-block:: python

    callback.run(hookpoint='on_hookpoint')


*************************************
Callback Pipeline
*************************************

In many cases, we may have to invoke multiple callbacks for a certain hookpoint.

Cinnamon adopts the nesting paradigm for callbacks as well since a ``Callback`` is a ``Component``.

A simple way to wrap multiple callbacks into a single parent ``Callback`` and execute them sequentially is the ``CallbackPipeline`` (see :ref:`pipeline` for more details about pipelines).

.. code-block:: python

    class CallbackPipeline(Pipeline, Callback):

        def setup(
                self,
                component: Component,
                save_path: Path
        ):
            callbacks = self.get_pipeline()
            for callback in callbacks:
                callback.setup(component=component,
                               save_path=save_path)

        def run(
                self,
                hookpoint: Optional[str] = None,
                logs: Dict[str, Any] = None
        ):
            components = self.get_pipeline()
            for component in components:
                component.run(hookpoint=hookpoint, logs=logs)


The ``setup`` method is usually used to attach a ``Component`` to a ``Callback`` (e.g., a ``Model`` instance).

The ``run`` method executes all child ``Callback`` components in sequential fashion for the specified hookpoint.

Thus, from a code point of view, the ``callbacks`` argument in a guarded function can either be a ``Callback`` or a ``CallbackPipeline``.