Installation
============

**Detect gibberish, garbled text, and corrupted content with high accuracy using advanced machine learning techniques.**

Installation from PyPI
----------------------

The easiest way to install pygarble is using pip:

.. code-block:: bash

   pip install pygarble

This will install the latest stable version along with all required dependencies.

Development Installation
-------------------------

To install pygarble for development:

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/brightertiger/pygarble.git
   cd pygarble

2. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements-dev.txt

4. Install the package in development mode:

.. code-block:: bash

   pip install -e .

Dependencies
------------

pygarble requires:

- Python 3.8+
- fasttext-wheel>=0.9.2 (for language detection strategy)

Optional dependencies for development:

- pytest (for testing)
- black (for code formatting)
- flake8 (for linting)
- mypy (for type checking)
- sphinx (for documentation)

Verifying Installation
----------------------

To verify that pygarble is installed correctly:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Test basic functionality
   detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
   result = detector.predict("test")
   print(f"Installation successful: {result}")

Language Detection Model
------------------------

The Language Detection strategy requires a FastText model. By default, pygarble will download the model automatically on first use. For offline environments or custom models:

.. code-block:: python

   detector = GarbleDetector(
       Strategy.LANGUAGE_DETECTION,
       model_path='/path/to/custom/model.bin'
   )

The default model is approximately 126MB and will be cached after the first download.
