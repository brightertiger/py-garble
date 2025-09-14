# pygarble

**Detect gibberish, garbled text, and corrupted content with high accuracy using advanced machine learning techniques.**

pygarble is a powerful Python library designed to identify nonsensical, garbled, or corrupted text content that often appears in data processing pipelines, user inputs, or automated systems. Whether you're dealing with random character sequences, encoding errors, keyboard mashing, or corrupted data streams, pygarble provides multiple detection strategies to filter out unwanted content and maintain data quality. The library uses statistical analysis, entropy calculations, pattern matching, and language detection to distinguish between meaningful text and gibberish with configurable sensitivity levels.

## Features

- **Multiple Detection Strategies**: Choose from 6 different garble detection algorithms
- **Scikit-learn Interface**: Familiar `predict()` and `predict_proba()` methods
- **Configurable Thresholds**: Adjust sensitivity for each strategy
- **Probability Scores**: Get confidence scores for garble detection
- **Modular Design**: Easy to extend with new detection strategies
- **Enterprise Ready**: Support for offline model paths and restricted environments
- **Smart Edge Cases**: Automatically detects extremely long strings without any whitespace (like base64 data)

## Installation

You can install pygarble using pip:

```bash
pip install pygarble
```

## Quick Start

```python
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
```

## Detection Strategies

Each strategy implements a different approach to detect garbled text. All strategies return probability scores between 0.0 and 1.0, where higher scores indicate more likely garbled text.

**Note**: For sentence and paragraph analysis, Character Frequency, Statistical Analysis, and Entropy Based strategies exclude whitespace characters (spaces, tabs, newlines) from their calculations since whitespace is structural, not content. This ensures consistent analysis regardless of formatting differences.

### 1. Character Frequency (`CHARACTER_FREQUENCY`)

**Implementation Logic**: Analyzes the frequency distribution of alphabetic characters in the text (excluding whitespace). Garbled text often has unusual character frequency patterns (e.g., too many repeated characters or skewed distributions).

**Algorithm**:
1. Count frequency of each alphabetic character (ignoring whitespace, numbers, and special characters)
2. Calculate the maximum character frequency ratio
3. Compare against threshold to determine if text is garbled

**Parameters**:
- `frequency_threshold` (float, default: 0.1): Maximum allowed character frequency ratio

```python
detector = GarbleDetector(
    Strategy.CHARACTER_FREQUENCY,
    threshold=0.3,
    frequency_threshold=0.1  # Character frequency threshold
)

# Examples
detector.predict("aaaaaaa")      # True - high frequency of 'a'
detector.predict("normal text")  # False - balanced character distribution
detector.predict("aaa aaa aaa")  # True - same as above (whitespace ignored)
```

### 2. Word Length (`WORD_LENGTH`)

**Implementation Logic**: Analyzes the average word length in the text. Garbled text often contains unusually long words or lacks proper word boundaries.

**Algorithm**:
1. Split text into words (whitespace-separated)
2. Calculate average word length
3. Compare against maximum allowed word length

**Parameters**:
- `max_word_length` (int, default: 20): Maximum allowed average word length

```python
detector = GarbleDetector(
    Strategy.WORD_LENGTH,
    threshold=0.5,
    max_word_length=20  # Maximum average word length
)

# Examples
detector.predict("supercalifragilisticexpialidocious")  # True - very long word
detector.predict("short words")                          # False - normal word lengths
```

### 3. Pattern Matching (`PATTERN_MATCHING`)

**Implementation Logic**: Uses configurable regex patterns to detect suspicious text patterns. Highly customizable with named patterns and override capabilities.

**Algorithm**:
1. Apply each configured regex pattern to the text
2. Count number of patterns that match
3. Calculate probability as ratio of matches to total patterns

**Default Patterns**:
- `special_chars`: Sequences of 3+ special characters `[^a-zA-Z0-9\s]{3,}`
- `repeated_chars`: Characters repeated 5+ times `(.)\1{4,}`
- `uppercase_sequence`: 5+ consecutive uppercase letters `[A-Z]{5,}`
- `long_numbers`: 8+ consecutive digits `[0-9]{8,}`

**Parameters**:
- `patterns` (dict, optional): Custom patterns dictionary `{name: regex_pattern}`
- `override_defaults` (bool, default: False): Whether to skip default patterns

```python
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

# Examples
detector.predict("AAAAA")              # True - uppercase sequence
detector.predict("@@@@@")              # True - special characters
detector.predict("test@example.com")   # True - email pattern (if configured)
detector.predict("normal text")        # False - no patterns match
```

### 4. Statistical Analysis (`STATISTICAL_ANALYSIS`)

**Implementation Logic**: Analyzes the ratio of alphabetic characters to content characters (excluding whitespace). Garbled text often has low alphabetic character ratios.

**Algorithm**:
1. Count alphabetic characters in the text
2. Count content characters (excluding whitespace)
3. Calculate ratio of alphabetic to content characters
4. Compare against minimum threshold

**Parameters**:
- `alpha_threshold` (float, default: 0.5): Minimum required alphabetic character ratio

```python
detector = GarbleDetector(
    Strategy.STATISTICAL_ANALYSIS,
    threshold=0.5,
    alpha_threshold=0.5  # Minimum alphabetic character ratio
)

# Examples
detector.predict("123456789")    # True - low alphabetic ratio
detector.predict("normal text")  # False - high alphabetic ratio
detector.predict("123 456 789")  # True - same as above (whitespace ignored)
```

### 5. Entropy Based (`ENTROPY_BASED`)

**Implementation Logic**: Uses Shannon entropy to measure alphabetic character diversity (excluding whitespace). Garbled text often has low entropy due to repetitive patterns.

**Algorithm**:
1. Calculate alphabetic character frequency distribution (ignoring whitespace, numbers, and special characters)
2. Compute Shannon entropy: H = -Σ(p_i * log2(p_i))
3. Compare against minimum entropy threshold

**Parameters**:
- `entropy_threshold` (float, default: 3.0): Minimum required entropy value

```python
detector = GarbleDetector(
    Strategy.ENTROPY_BASED,
    threshold=0.5,
    entropy_threshold=3.0  # Minimum entropy threshold
)

# Examples
detector.predict("aaaaaaa")      # True - low entropy (repetitive)
detector.predict("normal text")  # False - high entropy (diverse)
detector.predict("aaa aaa aaa")  # True - same as above (whitespace ignored)
```

### 6. Language Detection (`LANGUAGE_DETECTION`)

**Implementation Logic**: Uses FastText language identification to detect if text is in the expected language. Garbled text often fails language detection.

**Algorithm**:
1. Load FastText language identification model
2. Predict language probabilities for the text
3. Check if target language probability is above threshold

**Parameters**:
- `target_language` (str, default: 'en'): Expected language code
- `model_path` (str, optional): Path to custom FastText model

```python
detector = GarbleDetector(
    Strategy.LANGUAGE_DETECTION,
    threshold=0.5,
    target_language='en',  # Target language code
    model_path='/path/to/model.bin'  # Optional custom model path
)

# Examples
detector.predict("Hello world")           # False - detected as English
detector.predict("asdfghjkl")             # True - not detected as English
detector.predict("Bonjour le monde")     # True - detected as French, not English
```

## Advanced Usage

### Pattern Matching Configuration

The Pattern Matching strategy offers extensive configurability for custom use cases:

```python
# 1. Default patterns only
detector = GarbleDetector(Strategy.PATTERN_MATCHING)

# 2. Add custom patterns to defaults
custom_patterns = {
    'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone_pattern': r'\d{3}-\d{3}-\d{4}',
    'url_pattern': r'https?://[^\s]+',
    'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'
}
detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns=custom_patterns)

# 3. Override specific default patterns
override_patterns = {
    'special_chars': r'[^a-zA-Z0-9\s]{5,}',  # More restrictive (5+ chars)
    'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
}
detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns=override_patterns)

# 4. Use only custom patterns (no defaults)
detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns=custom_patterns, override_defaults=True)

# 5. Disable all patterns (returns 0.0 probability for everything)
detector = GarbleDetector(Strategy.PATTERN_MATCHING, patterns={}, override_defaults=True)
```

### Custom Thresholds

```python
# Adjust sensitivity
detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.3)
# Lower threshold = more sensitive (detects more as garbled)
# Higher threshold = less sensitive (detects less as garbled)
```

### Batch Processing

```python
detector = GarbleDetector(Strategy.PATTERN_MATCHING)

# Process multiple texts at once
texts = ["normal text", "AAAAA", "123456789", "mixed text"]
predictions = detector.predict(texts)
probabilities = detector.predict_proba(texts)

for text, pred, prob in zip(texts, predictions, probabilities):
    status = "GARBLED" if pred else "NORMAL"
    print(f"{text:15} -> {status:7} (confidence: {prob:.3f})")
```

### Multithreaded Batch Processing

For very large datasets, use multithreading:

```python
from pygarble import GarbleDetector, Strategy
import time

# Large dataset
texts = ["normal text", "AAAAA", "123456789", "mixed text"] * 2000

# Single-threaded processing
detector_single = GarbleDetector(Strategy.LANGUAGE_DETECTION)
start_time = time.time()
predictions_single = detector_single.predict(texts)
single_time = time.time() - start_time

# Multithreaded processing
detector_multi = GarbleDetector(Strategy.LANGUAGE_DETECTION, threads=4)
start_time = time.time()
predictions_multi = detector_multi.predict(texts)
multi_time = time.time() - start_time

print(f"Single-threaded: {single_time:.3f}s")
print(f"Multithreaded:   {multi_time:.3f}s")
print(f"Speedup:         {single_time/multi_time:.2f}x")
print(f"Results match:   {predictions_single == predictions_multi}")
```

### Multithreaded Processing

For large datasets, you can enable multithreaded processing:

```python
# Enable multithreading for large datasets
detector = GarbleDetector(Strategy.LANGUAGE_DETECTION, threads=4)

# Process large batch with multiple threads
large_texts = ["text"] * 1000  # 1000 texts
predictions = detector.predict(large_texts)
```

**Note**: Multithreading is most beneficial for:
- Large datasets (100+ texts)
- I/O-bound strategies (Language Detection with model loading)
- Strategies with expensive computations

For small datasets or CPU-bound strategies, single-threaded processing may be faster due to threading overhead.

### Language Detection for Offline Environments

```python
# For environments without internet access
detector = GarbleDetector(
    Strategy.LANGUAGE_DETECTION,
    model_path='/offline/path/lid.176.bin',
    target_language='en',
    threshold=0.5
)
```

### Edge Case Handling

The package automatically handles common edge cases:

```python
# Extremely long strings without any whitespace (like base64 data)
detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, max_string_length=1000)

# These will be detected as garbled (no whitespace)
long_base64 = "SGVsbG9Xb3JsZEhlbGxvV29ybGRIZWxsb1dvcmxk" * 50
long_url = "https://example.com/" + "a" * 1000
long_no_whitespace = "a" * 1001

# These will NOT be detected as garbled (have whitespace)
long_paragraph = "This is a normal sentence. " * 50
long_with_tabs = "word\t" * 250
long_with_newlines = "word\n" * 250
long_mixed_whitespace = "word \t\n " * 200

print(detector.predict(long_base64))           # True
print(detector.predict(long_url))              # True  
print(detector.predict(long_no_whitespace))    # True
print(detector.predict(long_paragraph))       # False
print(detector.predict(long_with_tabs))       # False
print(detector.predict(long_with_newlines))    # False
print(detector.predict(long_mixed_whitespace)) # False
```

## API Reference

### GarbleDetector

```python
GarbleDetector(strategy: Strategy, threshold: float = 0.5, threads: int = None, **kwargs)
```

**Parameters:**
- `strategy`: Detection strategy to use
- `threshold`: Probability threshold for binary predictions (default: 0.5)
- `threads`: Number of threads for batch processing (default: None, single-threaded)
- `**kwargs`: Strategy-specific parameters

**Methods:**
- `predict(X)`: Returns binary predictions (True/False)
- `predict_proba(X)`: Returns probability scores (0.0-1.0)

### Strategy Enum

```python
class Strategy(Enum):
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    LANGUAGE_DETECTION = "language_detection"
```

## Examples

### Text Quality Filtering

```python
from pygarble import GarbleDetector, Strategy

# Filter out garbled text from a dataset
detector = GarbleDetector(Strategy.ENTROPY_BASED, threshold=0.7)

texts = [
    "This is normal English text",
    "asdfghjkl",
    "Hello world",
    "AAAAA",
    "Mixed content with numbers 123"
]

clean_texts = []
for text in texts:
    if not detector.predict(text):
        clean_texts.append(text)

print("Clean texts:", clean_texts)
```

### Multi-Strategy Ensemble

```python
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
```

### Language Validation

```python
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
```

### Custom Pattern Matching for Data Validation

```python
from pygarble import GarbleDetector, Strategy

# Create a detector for validating user input data
validation_patterns = {
    'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone_pattern': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
    'ssn_pattern': r'\d{3}-\d{2}-\d{4}',
    'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    'suspicious_chars': r'[^a-zA-Z0-9\s@.-]{3,}'  # Override default special_chars
}

detector = GarbleDetector(
    Strategy.PATTERN_MATCHING,
    patterns=validation_patterns,
    threshold=0.1  # Low threshold to catch any suspicious patterns
)

# Test various inputs
test_inputs = [
    "john.doe@example.com",      # Valid email
    "555-123-4567",             # Valid phone
    "123-45-6789",              # Valid SSN
    "4532 1234 5678 9012",      # Valid credit card
    "normal text",               # Normal text
    "asdfghjkl",                # Random characters
    "user@domain.com $$$",      # Email with suspicious chars
]

for text in test_inputs:
    proba = detector.predict_proba(text)
    is_suspicious = detector.predict(text)
    status = "SUSPICIOUS" if is_suspicious else "VALID"
    print(f"{text:25} -> {status:10} (confidence: {proba:.3f})")
```

### Domain-Specific Pattern Detection

```python
from pygarble import GarbleDetector, Strategy

# Create patterns for detecting code-like content
code_patterns = {
    'function_call': r'\w+\s*\([^)]*\)',
    'variable_assignment': r'\w+\s*=\s*\w+',
    'json_pattern': r'\{[^}]*"[^"]*"[^}]*\}',
    'sql_pattern': r'(SELECT|INSERT|UPDATE|DELETE)\s+.*FROM',
    'html_pattern': r'<[^>]+>',
    'long_identifiers': r'\w{20,}',  # Very long variable names
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
    "very_long_variable_name_that_exceeds_normal_length",
    "user_data = {'name': 'John', 'age': 30}",
]

for text in mixed_content:
    proba = detector.predict_proba(text)
    is_code_like = detector.predict(text)
    status = "CODE-LIKE" if is_code_like else "NORMAL"
    print(f"{text:50} -> {status:10} (confidence: {proba:.3f})")
```

## Dependencies

- Python 3.8+
- fasttext-wheel>=0.9.2 (for language detection strategy)

## Development

To set up a development environment:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/py-garble.git
cd py-garble
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Run tests:
```bash
pytest tests/
```

5. Run linting:
```bash
flake8 pygarble/
black pygarble/
mypy pygarble/
```

6. Run CI checks locally:
```bash
./scripts/test_ci.sh
```

### Continuous Integration

This project uses GitHub Actions for CI/CD. The workflow runs:

- **Tests**: All 44 tests across 5 test files on Python 3.8-3.12
- **Linting**: Code style checks with flake8, black, and isort
- **Type Checking**: Static analysis with mypy
- **Package Building**: Ensures the package builds successfully

For complete setup instructions, branch protection rules, and publishing guides, see [`.github/instructions.txt`](.github/instructions.txt).

## Architecture

The package uses a modular strategy pattern:

```
pygarble/
├── __init__.py
├── core.py                    # Main GarbleDetector class
└── strategies/
    ├── __init__.py
    ├── base.py                # Abstract base strategy
    ├── character_frequency.py
    ├── word_length.py
    ├── pattern_matching.py
    ├── statistical_analysis.py
    ├── entropy_based.py
    └── language_detection.py
```

Each strategy implements:
- `_predict_impl(text)`: Returns boolean prediction
- `_predict_proba_impl(text)`: Returns probability score

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Adding New Strategies

To add a new detection strategy:

1. Create a new file in `pygarble/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `_predict_impl()` and `_predict_proba_impl()`
4. Add the strategy to the enum and strategy map
5. Add tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### 0.1.1 (Upcoming)
- **Enhanced Pattern Matching Strategy**: Added configurable named patterns with override capabilities
- **Improved Whitespace Handling**: Updated Statistical Analysis to exclude whitespace from calculations for consistent sentence/paragraph analysis
- **Multithreaded Processing**: Added support for parallel processing of large datasets with configurable thread count
- **New Features**:
  - Named pattern dictionary with descriptive keys
  - `override_defaults` parameter to skip default patterns
  - Custom pattern merging with defaults
  - Pattern override functionality for specific defaults
  - `threads` parameter for multithreaded batch processing
- **Improved Documentation**: Added detailed implementation logic for all strategies and comprehensive documentation website
- **Backward Compatibility**: All existing code continues to work unchanged

### 0.1.0
- Initial release
- 6 detection strategies
- Scikit-learn-like interface
- Probability scoring
- Modular architecture
- Comprehensive test coverage
