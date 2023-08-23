.. _catalog:

Available ``Configuration``
*************************************

Currently, ``cinnamon-generic`` provides the following registered ``Configuration``.

-------------------
Calibrator
-------------------

- ``name='calibrator', tags={'grid'}, namespace='generic'``: the default ``GridSearchCalibrator``.
- ``name='calibrator', tags={'random'}, namespace='generic'``: the default ``RandomSearchCalibrator``.
- ``name='calibrator', tags={'hyperopt'}, namespace='generic'``: the default ``HyperOptCalibrator``.

-------------------
Data Splitter
-------------------

- ``name='data_splitter', tags={'tt', 'sklearn'}, namespace='generic'``: the default ``TTSplitter``.


--------------------
File Manager
--------------------

- ``name='file_manager', tags={'default'}, namespace='generic'``: the default ``FileManager``.

--------------------
Helper
--------------------

- ``name='helper', tags={'default'}, namespace='generic'``: the default ``Helper``.