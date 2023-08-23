.. _helper:

Framework Helper
*************************************

A ``Helper`` is a ``Component`` specialized in handling seeding and backend-specific behaviours.

This general ``Helper`` component expects an input seed to fix numpy and random packages stochasticity.

.. code-block:: python

    helper.run(seed=42)


***************************
Registered configurations
***************************

The ``cinnamon-generic`` package provides the following registered configurations:

- ``name='helper', tags={'default'}, namespace='generic'``: the default ``Helper``.