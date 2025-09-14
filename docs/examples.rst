Examples
========

This section provides practical examples of using pygarble for various tasks.

Text Filtering
--------------

Filter garbled text from a dataset:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Create a detector
   detector = GarbleDetector(Strategy.ENTROPY_BASED, threshold=0.7)

   # Sample dataset
   texts = [
       "This is normal English text",
       "asdfghjkl",
       "Hello world",
       "AAAAA",
       "Mixed content with numbers 123"
   ]

   # Filter out garbled text
   clean_texts = []
   for text in texts:
       if not detector.predict(text):
           clean_texts.append(text)

   print("Clean texts:", clean_texts)

Multi-Strategy Ensemble
-----------------------

Combine multiple strategies for better detection:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   strategies = [
       Strategy.CHARACTER_FREQUENCY,
       Strategy.PATTERN_MATCHING,
       Strategy.ENTROPY_BASED
   ]

   text = "suspicious text here"
   votes = 0

   for strategy in strategies:
       detector = GarbleDetector(strategy, threshold=0.5)
       if detector.predict(text):
           votes += 1

   # Text is considered garbled if majority of strategies agree
   is_garbled = votes > len(strategies) // 2
   print(f"Text is garbled: {is_garbled}")

Language Validation
-------------------

Validate that text is in the expected language:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Validate that text is in English
   detector = GarbleDetector(
       Strategy.LANGUAGE_DETECTION,
       target_language='en',
       threshold=0.3
   )

   user_inputs = [
       "Hello, how are you?",
       "Bonjour, comment allez-vous?",
       "asdfghjkl",
       "123456789"
   ]

   for text in user_inputs:
       proba = detector.predict_proba(text)
       is_english = not detector.predict(text)
       print(f"{text:25} -> English: {is_english:5} (confidence: {1-proba:.3f})")

Pattern Customization
---------------------

Customize pattern matching for specific use cases:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Create patterns for data validation
   validation_patterns = {
       'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
       'phone_pattern': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
       'ssn_pattern': r'\d{3}-\d{2}-\d{4}',
       'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
   }

   detector = GarbleDetector(
       Strategy.PATTERN_MATCHING,
       patterns=validation_patterns,
       threshold=0.1
   )

   # Test various inputs
   test_inputs = [
       "john.doe@example.com",      # Valid email
       "555-123-4567",             # Valid phone
       "123-45-6789",              # Valid SSN
       "4532 1234 5678 9012",      # Valid credit card
       "normal text",               # Normal text
       "asdfghjkl",                # Random characters
   ]

   for text in test_inputs:
       proba = detector.predict_proba(text)
       is_suspicious = detector.predict(text)
       status = "SUSPICIOUS" if is_suspicious else "VALID"
       print(f"{text:25} -> {status:10} (confidence: {proba:.3f})")

Domain-Specific Detection
-------------------------

Detect code-like content:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Create patterns for detecting code-like content
   code_patterns = {
       'function_call': r'\w+\s*\([^)]*\)',
       'variable_assignment': r'\w+\s*=\s*\w+',
       'json_pattern': r'\{[^}]*"[^"]*"[^}]*\}',
       'sql_pattern': r'(SELECT|INSERT|UPDATE|DELETE)\s+.*FROM',
       'html_pattern': r'<[^>]+>',
   }

   # Use only custom patterns (no defaults)
   detector = GarbleDetector(
       Strategy.PATTERN_MATCHING,
       patterns=code_patterns,
       override_defaults=True,
       threshold=0.2
   )

   # Test mixed content
   mixed_content = [
       "This is normal text",
       "def calculate_total(items):",
       "SELECT * FROM users WHERE id = 1",
       "<div class='container'>Hello</div>",
       "user_data = {'name': 'John', 'age': 30}",
   ]

   for text in mixed_content:
       proba = detector.predict_proba(text)
       is_code_like = detector.predict(text)
       status = "CODE-LIKE" if is_code_like else "NORMAL"
       print(f"{text:50} -> {status:10} (confidence: {proba:.3f})")

Batch Processing
----------------

Process large datasets efficiently:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   detector = GarbleDetector(Strategy.PATTERN_MATCHING)

   # Large dataset
   texts = ["normal text", "AAAAA", "123456789", "mixed text"] * 1000

   # Process all at once
   predictions = detector.predict(texts)
   probabilities = detector.predict_proba(texts)

   # Count results
   garbled_count = sum(predictions)
   total_count = len(predictions)
   
   print(f"Found {garbled_count} garbled texts out of {total_count} total")
   print(f"Garbled percentage: {garbled_count/total_count*100:.1f}%")

Threshold Tuning
----------------

Find the optimal threshold for your use case:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Test data with known labels
   test_data = [
       ("normal text", False),
       ("aaaaaaa", True),
       ("Hello world", False),
       ("asdfghjkl", True),
       ("123456789", True),
   ]

   # Test different thresholds
   thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
   
   for threshold in thresholds:
       detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=threshold)
       
       correct = 0
       for text, expected in test_data:
           predicted = detector.predict(text)
           if predicted == expected:
               correct += 1
       
       accuracy = correct / len(test_data)
       print(f"Threshold {threshold}: {accuracy:.1%} accuracy")
