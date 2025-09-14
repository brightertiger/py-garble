Detection Strategies
====================

pygarble provides 7 different strategies for detecting garbled text. Each strategy implements a different approach and is suitable for different types of garbled text.

Overview
--------

All strategies return probability scores between 0.0 and 1.0, where higher scores indicate more likely garbled text. The strategies exclude whitespace characters from their calculations for consistent sentence and paragraph analysis.

Character Frequency Strategy
-----------------------------

**Purpose**: Detects text with unusual character frequency patterns.

**Implementation**: Analyzes the frequency distribution of alphabetic characters (excluding whitespace). Garbled text often has repetitive characters or skewed distributions.

**Algorithm**:
1. Count frequency of each alphabetic character
2. Calculate the maximum character frequency ratio
3. Compare against threshold

**Parameters**:
- ``frequency_threshold`` (float, default: 0.1): Maximum allowed character frequency ratio

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, frequency_threshold=0.1)
   
   detector.predict("aaaaaaa")      # True - high frequency of 'a'
   detector.predict("normal text")   # False - balanced distribution
   detector.predict("aaa aaa aaa")  # True - same as above (whitespace ignored)

Word Length Strategy
--------------------

**Purpose**: Identifies text with unusually long words.

**Implementation**: Analyzes the average word length in the text. Garbled text often contains unusually long words or lacks proper word boundaries.

**Algorithm**:
1. Split text into words (whitespace-separated)
2. Calculate average word length
3. Compare against maximum allowed word length

**Parameters**:
- ``max_word_length`` (int, default: 20): Maximum allowed average word length

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.WORD_LENGTH, max_word_length=20)
   
   detector.predict("supercalifragilisticexpialidocious")  # True - very long word
   detector.predict("short words")                          # False - normal lengths

Pattern Matching Strategy
-------------------------

**Purpose**: Uses configurable regex patterns to detect suspicious text patterns.

**Implementation**: Highly customizable with named patterns and override capabilities. Applies each configured regex pattern and calculates probability as ratio of matches to total patterns.

**Default Patterns**:
- ``special_chars``: Sequences of 3+ special characters ``[^a-zA-Z0-9\s]{3,}``
- ``repeated_chars``: Characters repeated 5+ times ``(.)\1{4,}``
- ``uppercase_sequence``: 5+ consecutive uppercase letters ``[A-Z]{5,}``
- ``long_numbers``: 8+ consecutive digits ``[0-9]{8,}``

**Parameters**:
- ``patterns`` (dict, optional): Custom patterns dictionary ``{name: regex_pattern}``
- ``override_defaults`` (bool, default: False): Whether to skip default patterns

**Example**:

.. code-block:: python

   # Default patterns only
   detector = GarbleDetector(Strategy.PATTERN_MATCHING)
   
   # Custom patterns added to defaults
   custom_patterns = {
       'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
       'phone_pattern': r'\d{3}-\d{3}-\d{4}'
   }
   detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns=custom_patterns)
   
   # Override defaults completely
   detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns=custom_patterns, override_defaults=True)

Statistical Analysis Strategy
-----------------------------

**Purpose**: Analyzes the ratio of alphabetic characters to content characters.

**Implementation**: Counts alphabetic characters and content characters (excluding whitespace), then calculates their ratio. Garbled text often has low alphabetic character ratios.

**Algorithm**:
1. Count alphabetic characters in the text
2. Count content characters (excluding whitespace)
3. Calculate ratio of alphabetic to content characters
4. Compare against minimum threshold

**Parameters**:
- ``alpha_threshold`` (float, default: 0.5): Minimum required alphabetic character ratio

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.STATISTICAL_ANALYSIS, alpha_threshold=0.5)
   
   detector.predict("123456789")    # True - low alphabetic ratio
   detector.predict("normal text")  # False - high alphabetic ratio
   detector.predict("123 456 789")  # True - same as above (whitespace ignored)

Entropy Based Strategy
----------------------

**Purpose**: Uses Shannon entropy to measure character diversity.

**Implementation**: Calculates Shannon entropy of alphabetic character distribution (excluding whitespace). Garbled text often has low entropy due to repetitive patterns.

**Algorithm**:
1. Calculate alphabetic character frequency distribution
2. Compute Shannon entropy: H = -Σ(p_i * log2(p_i))
3. Compare against minimum entropy threshold

**Parameters**:
- ``entropy_threshold`` (float, default: 3.0): Minimum required entropy value

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.ENTROPY_BASED, entropy_threshold=3.0)
   
   detector.predict("aaaaaaa")      # True - low entropy (repetitive)
   detector.predict("normal text")  # False - high entropy (diverse)
   detector.predict("aaa aaa aaa")  # True - same as above (whitespace ignored)

Language Detection Strategy
---------------------------

**Purpose**: Uses FastText language identification to detect if text is in the expected language.

**Implementation**: Loads a FastText language identification model and predicts language probabilities. Garbled text often fails language detection.

**Algorithm**:
1. Load FastText language identification model
2. Predict language probabilities for the text
3. Check if target language probability is above threshold

**Parameters**:
- ``target_language`` (str, default: 'en'): Expected language code
- ``model_path`` (str, optional): Path to custom FastText model

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.LANGUAGE_DETECTION, target_language='en')
   
   detector.predict("Hello world")           # False - detected as English
   detector.predict("asdfghjkl")             # True - not detected as English
   detector.predict("Bonjour le monde")     # True - detected as French, not English

English Word Validation Strategy
--------------------------------

**Purpose**: Tokenizes text and validates words against an English dictionary using pyspellchecker.

**Implementation**: Uses pyspellchecker to check each word against an English dictionary. Garbled text often contains many invalid English words.

**Algorithm**:
1. Tokenize text into individual words (alphabetic characters only)
2. Check each word against English dictionary
3. Calculate ratio of valid words to total words
4. Compare against threshold to determine if text is garbled

**Parameters**:
- ``valid_word_threshold`` (float, default: 0.7): Minimum required ratio of valid English words

**Example**:

.. code-block:: python

   detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, valid_word_threshold=0.7)
   
   detector.predict("hello world this is normal text")  # False - all words are valid
   detector.predict("asdfghjkl qwertyuiop zxcvbnm")    # True - no valid words
   detector.predict("hello asdfgh world qwerty")        # False - 50% valid words (above threshold)

Choosing the Right Strategy
---------------------------

**Character Frequency**: Best for detecting repetitive text patterns (e.g., "aaaaaaa", "asdfghjkl")

**Word Length**: Best for detecting text with unusually long words or lack of word boundaries

**Pattern Matching**: Most flexible - can be customized for specific patterns (emails, phone numbers, etc.)

**Statistical Analysis**: Best for detecting text with unusual character composition (mostly numbers, special characters)

**Entropy Based**: Best for detecting text with low character diversity (repetitive patterns)

**Language Detection**: Best for detecting text that's not in the expected language

**English Word Validation**: Best for detecting text with many invalid English words (gibberish, typos, non-English words)

For best results, consider using multiple strategies in combination. See :doc:`examples` for examples.
