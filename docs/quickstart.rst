Quick Start Guide
=================

This guide will get you up and running with pygarble in just a few minutes.

Basic Usage
-----------

The simplest way to use pygarble is to create a detector and call its methods:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Create a detector
   detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)

   # Detect if text is garbled
   is_garbled = detector.predict("aaaaaaa")
   print(is_garbled)  # True

   # Get probability score
   probability = detector.predict_proba("aaaaaaa")
   print(probability)  # 1.0

Batch Processing
-----------------

Process multiple texts at once:

.. code-block:: python

   texts = ["normal text", "aaaaaaa", "asdfghjkl", "Hello world"]
   
   # Get binary predictions
   predictions = detector.predict(texts)
   print(predictions)  # [False, True, True, False]
   
   # Get probability scores
   probabilities = detector.predict_proba(texts)
   print(probabilities)  # [0.2, 1.0, 0.8, 0.1]

Choosing a Strategy
-------------------

pygarble offers 6 different detection strategies. Here's a quick comparison:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   text = "suspicious text here"

   # Character Frequency - detects repetitive characters
   detector1 = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
   
   # Word Length - detects unusually long words
   detector2 = GarbleDetector(Strategy.WORD_LENGTH)
   
   # Pattern Matching - uses regex patterns (highly configurable)
   detector3 = GarbleDetector(Strategy.PATTERN_MATCHING)
   
   # Statistical Analysis - analyzes character ratios
   detector4 = GarbleDetector(Strategy.STATISTICAL_ANALYSIS)
   
   # Entropy Based - measures character diversity
   detector5 = GarbleDetector(Strategy.ENTROPY_BASED)
   
   # Language Detection - checks if text is in expected language
   detector6 = GarbleDetector(Strategy.LANGUAGE_DETECTION)

   # Test all strategies
   for i, detector in enumerate([detector1, detector2, detector3, detector4, detector5, detector6], 1):
       result = detector.predict(text)
       print(f"Strategy {i}: {result}")

Adjusting Sensitivity
---------------------

Control how sensitive the detector is by adjusting the threshold:

.. code-block:: python

   # More sensitive (detects more as garbled)
   sensitive_detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.3)
   
   # Less sensitive (detects less as garbled)
   conservative_detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.7)

   text = "borderline case"
   print(f"Sensitive: {sensitive_detector.predict(text)}")      # True
   print(f"Conservative: {conservative_detector.predict(text)}") # False

Strategy-Specific Parameters
----------------------------

Each strategy has its own parameters for fine-tuning:

.. code-block:: python

   # Character Frequency with custom threshold
   detector = GarbleDetector(
       Strategy.CHARACTER_FREQUENCY,
       frequency_threshold=0.2  # Higher threshold = more sensitive
   )

   # Word Length with custom max length
   detector = GarbleDetector(
       Strategy.WORD_LENGTH,
       max_word_length=15  # Words longer than 15 chars are suspicious
   )

   # Pattern Matching with custom patterns
   detector = GarbleDetector(
       Strategy.PATTERN_MATCHING,
       patterns={
           'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
           'phone_pattern': r'\d{3}-\d{3}-\d{4}'
       }
   )

   # Language Detection for specific language
   detector = GarbleDetector(
       Strategy.LANGUAGE_DETECTION,
       target_language='en',  # Expect English text
       threshold=0.3
   )

Multithreaded Processing
------------------------

For large datasets, you can enable multithreaded processing:

.. code-block:: python

   # Enable multithreading for large datasets
   detector = GarbleDetector(Strategy.LANGUAGE_DETECTION, threads=4)

   # Process large batch with multiple threads
   large_texts = ["text"] * 1000  # 1000 texts
   predictions = detector.predict(large_texts)

**Note**: Multithreading is most beneficial for:
- Large datasets (100+ texts)
- I/O-bound strategies (Language Detection with model loading)
- Strategies with expensive computations

For small datasets or CPU-bound strategies, single-threaded processing may be faster due to threading overhead.

Next Steps
----------

- Learn about each strategy in detail: :doc:`strategies`
- See practical examples: :doc:`examples`
- Explore the full API: :doc:`api`
