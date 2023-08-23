# Cinnamon Generic 

The generic package offers several ``Component`` and related ``Configuration`` for machine-learning.

Additionally, it provides the first set of **commands**: high-level APIs for speeding up cinnamon registration.

## Components and Configurations


The generic package defines ``Component`` and ``Configuration`` for

- Hyper-parameter calibration
- Callbacks
- Data loaders
- Data splitters (e.g., train, validation and test)
- File manager
- Framework helper (e.g., for deterministic runs)
- Metrics
- Model API
- Pipeline API (i.e., a special nested ``Component`` that executes children sequentially)
- Data processors
- Model validation routines: train and test, cross-validation

## Commands

Commands are syntactic sugar for several cinnamon operations.

For instance, the ``setup_registry()`` performs (i) custom module loading; (ii) registration DAG evaluation and (iii) registration.
Thus, it is just necessary to invoke this command at the beginning of each runnable script to use cinnamon.

Other commands are more specific to the generic package:

- Running a generic ``Component``.
- Running a ``Routine`` component in **training** and **inference** modes.
- Running multiple ``Component`` in sequential fashion.
- Running multiple ``Routine`` components in sequential fashion.
- Running a ``Calibrator`` component.

## Install

pip

      pip install cinnamon-generic

git

      git clone https://github.com/federicoruggeri/cinnamon_generic

## Contribute

Want to contribute with new ``Component`` and ``Configuration``?

Feel free to submit a merge request! 

Cinnamon is meant to be a community project :)

## Contact

Don't hesitate to contact:
- Federico Ruggeri @ [federico.ruggeri6@unibo.it](mailto:federico.ruggeri6@unibo.it)

for questions/doubts/issues!