from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.pipeline import Pipeline


class Callback(Component):
    """
    Generic ``Callback`` component.
    A ``Callback`` component defines execution flow hookpoints for flow customization and side effects.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(Callback, self).__init__(**kwargs)
        self.component: Optional[Component] = None
        self.save_path: Optional[Path] = None

    def setup(
            self,
            component: Component,
            save_path: Path
    ):
        """
        Set-ups the ``Callback`` instance with a ``Component`` reference for quick attributes access and
        serialization save path.

        Args:
            component: a ``Component`` instance that exposes hookpoints for this ``Callback``
            save_path: path where to potentially save ``Callback`` side effects.
        """

        self.component = component
        self.save_path = save_path

    def run(
            self,
            hookpoint: Optional[str] = None,
            logs: Dict[str, Any] = None
    ):
        """
        Runs the ``Callback``'s specific hookpoint.
        If the ``Callback`` doesn't have the specified hookpoint, nothing happens.

        Args:
            hookpoint: name of the hookpoint method to invoke
            logs: optional arguments for the hookpoint
        """

        if hasattr(self, hookpoint):
            hookpoint_method = getattr(self, hookpoint)
            hookpoint_method(logs=logs)


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


def hookpoint_guard(
        func: Callable,
        hookpoint: Optional[str] = None
):
    """
    A decorator to enable ``Callback``'s hookpointing
    Args:
        func: the function to be decorated
        hookpoint: the ``Callback``'s base hookpoint to be invoked. In particular, two different hookpoints
        will be called
        - *hookpoint*_begin
        - *hookpoint*_end

        For instance, if ``hookpoint = 'on_fit'``, the following execution flow is considered:
        - ``Callback.on_fit_begin(...)``
        - func(...)
        - ``Callback._on_fit_end(...)

    Returns:
        The decorated method with specified ``Callback``'s hookpoint.
    """

    hookpoint = hookpoint if hookpoint is not None else func.__name__
    start_hookpoint = f'on_{hookpoint}_begin'
    end_hookpoint = f'on_{hookpoint}_end'

    def func_wrap(
            *args,
            **kwargs
    ):
        callbacks = kwargs.get('callbacks', None)

        if callbacks is not None:
            callbacks.run(hookpoint=start_hookpoint)

        res = func(*args, **kwargs)

        if callbacks is not None:
            callbacks.run(hookpoint=end_hookpoint, logs=res)

        return res

    return func_wrap


def guard(
        hookpoint: Optional[str] = None
):
    """
    A decorator to mark a ``Callback`` hookpoints.

    Args:
        hookpoint: the ``Callback``'s base hookpoint to be invoked. In particular, two different hookpoints
        will be called
        - *hookpoint*_begin
        - *hookpoint*_end

        For instance, if ``hookpoint = 'on_fit'``, the following execution flow is considered:
        - ``Callback.on_fit_begin(...)``
        - func(...)
        - ``Callback._on_fit_end(...)

    Returns:
        The decorated method with specified ``Callback``'s hookpoint.
    """

    return partial(hookpoint_guard, hookpoint=hookpoint)


__all__ = ['Callback', 'CallbackPipeline', 'guard']
