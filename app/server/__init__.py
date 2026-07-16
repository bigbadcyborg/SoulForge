"""Local HTTP/WebSocket API exposing SoulForge to the desktop GUI front end.

The server runs inside WSL alongside the model and wraps the UI-agnostic
``ChatController``. The native Windows GUI (``gui/``) talks to it over
localhost, which WSL2 forwards to the Windows host automatically.
"""
