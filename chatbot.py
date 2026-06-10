"""Legacy entry point.

The chatbot has been refactored into the ``app`` package (Iteration 1).
This shim keeps ``python chatbot.py`` working; prefer ``python -m app.main``.
"""

from app.main import main

if __name__ == "__main__":
    main()
