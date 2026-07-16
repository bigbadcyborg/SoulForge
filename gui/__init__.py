"""Native Windows desktop GUI for SoulForge.

Runs as a separate process on the Windows host (its own venv,
``gui/requirements-windows.txt``) and talks to the WSL API server
(``app/server``) over localhost. Owns the window, command buttons, global
hotkeys, screen snapshot, and microphone capture; all inference happens in WSL.
"""
