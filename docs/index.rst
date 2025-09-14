pygarble Documentation
========================

A Python package for detecting garbled text using multiple detection strategies with a scikit-learn-like interface.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   strategies
   api
   examples
   contributing

Features
--------

- **Multiple Detection Strategies**: Choose from 6 different garble detection algorithms
- **Scikit-learn Interface**: Familiar `predict()` and `predict_proba()` methods
- **Configurable Thresholds**: Adjust sensitivity for each strategy
- **Probability Scores**: Get confidence scores for garble detection
- **Modular Design**: Easy to extend with new detection strategies
- **Enterprise Ready**: Support for offline model paths and restricted environments
- **Smart Edge Cases**: Automatically detects extremely long strings without any whitespace

Quick Start
-----------

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Create a detector with character frequency strategy
   detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.5)

   # Detect garbled text
   texts = ["normal text", "aaaaaaa", "asdfghjkl"]
   results = detector.predict(texts)
   print(results)  # [False, True, True]

   # Get probability scores
   probabilities = detector.predict_proba(texts)
   print(probabilities)  # [0.2, 1.0, 0.8]

Detection Strategies
--------------------

pygarble provides 6 different strategies for detecting garbled text:

1. **Character Frequency**: Detects unusual character frequency patterns
2. **Word Length**: Identifies text with unusually long words
3. **Pattern Matching**: Uses configurable regex patterns (highly customizable)
4. **Statistical Analysis**: Analyzes alphabetic character ratios
5. **Entropy Based**: Uses Shannon entropy to measure character diversity
6. **Language Detection**: Uses FastText for language identification

For detailed information about each strategy, see :doc:`strategies`.

Installation
------------

.. code-block:: bash

   pip install pygarble

For development installation, see :doc:`installation`.

Examples
--------

See the :doc:`examples` section for practical examples including:
- Text filtering from datasets
- Multi-strategy ensemble approaches
- Language validation
- Pattern customization

API Reference
-------------

See the :doc:`api` section for complete API documentation including:
- Main GarbleDetector class
- Strategy implementations
- Strategy enumeration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`