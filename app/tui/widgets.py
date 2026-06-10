"""Reusable Textual widgets for the SoulForge TUI."""

from __future__ import annotations

from rich.text import Text
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, Static

from app.core.feature_state import FEATURE_KEYS

ROLE_LABELS = {
    "user": "You",
    "assistant": "Bot",
    "system": "System",
}

ROLE_STYLES = {
    "user": "bold cyan",
    "assistant": "bold green",
    "system": "bold yellow",
}


class ChatMessage(Static):
    """A single chat bubble; supports incremental token appends for streaming."""

    def __init__(self, role: str, text: str = "") -> None:
        self.role = role
        self._text = text
        super().__init__(self._build(), classes=f"role-{role}")

    def append(self, chunk: str) -> None:
        self._text += chunk
        self.update(self._build())

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(self._build())

    def _build(self) -> Text:
        label = ROLE_LABELS.get(self.role, self.role)
        style = ROLE_STYLES.get(self.role, "bold white")
        renderable = Text()
        renderable.append(f"{label}\n", style=style)
        renderable.append(self._text or "")
        return renderable


class StatusBar(Static):
    """Bottom status bar showing model name, active features, and state."""

    def __init__(self) -> None:
        self._model = "—"
        self._features = "—"
        self._state = "Starting"
        super().__init__(self._build())

    def set_model(self, model: str) -> None:
        self._model = model
        self.update(self._build())

    def set_features(self, features: str) -> None:
        self._features = features
        self.update(self._build())

    def set_state(self, state: str) -> None:
        self._state = state
        self.update(self._build())

    def _build(self) -> Text:
        text = Text()
        text.append(" model: ", style="dim")
        text.append(self._model, style="bold")
        text.append("  │  features: ", style="dim")
        text.append(self._features, style="bold")
        text.append("  │  ", style="dim")
        text.append(self._state, style="bold magenta")
        return text


class RagSelectionModal(ModalScreen):
    """Modal for selecting RAG documents."""

    def __init__(self, available_sources: list[str]) -> None:
        super().__init__()
        self.available_sources = available_sources
        # Map sanitized IDs back to original source names
        self.id_to_source = {
            self._sanitize_id(source): source for source in available_sources
        }
        self.selected_sources: list[str] = []

    @staticmethod
    def _sanitize_id(source: str) -> str:
        """Convert source name to valid Textual ID (alphanumeric, underscore, hyphen only)."""
        # Replace invalid characters with underscores
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
        return f"doc_{sanitized}"

    def compose(self):
        """Compose the modal UI."""
        with Vertical(id="rag-modal-container"):
            yield Label("Select documents to use for RAG:")
            yield Checkbox("All documents", id="all-checkbox")
            for source in self.available_sources:
                sanitized_id = self._sanitize_id(source)
                yield Checkbox(f"  {source}", id=sanitized_id)
            with Container(id="button-container"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "ok-button":
            # Collect selected documents
            all_checkbox = self.query_one("#all-checkbox", Checkbox)
            if all_checkbox.value:
                # All documents selected
                self.dismiss(self.available_sources)
            else:
                selected = []
                for sanitized_id, source in self.id_to_source.items():
                    checkbox = self.query_one(f"#{sanitized_id}", Checkbox)
                    if checkbox.value:
                        selected.append(source)
                self.dismiss(selected if selected else None)
        elif event.button.id == "cancel-button":
            self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "all-checkbox":
            # Toggle all other checkboxes based on "All" state
            for sanitized_id in self.id_to_source.keys():
                checkbox = self.query_one(f"#{sanitized_id}", Checkbox)
                checkbox.value = event.checkbox.value


FEATURE_TOGGLE_LABELS: dict[str, str] = {
    "soul": "SOUL (persona)",
    "rag": "RAG",
    "memory": "Memory",
    "skills": "Skills",
    "curator": "Curator",
    "kanban": "Kanban",
    "show_sources": "Show sources",
    "streaming": "Streaming",
}

FEATURE_TOGGLE_DESCRIPTIONS: dict[str, str] = {
    "soul": "Inject SOUL.md persona, tone, and behavior into the system prompt.",
    "rag": "Retrieve relevant chunks from indexed docs and ground replies in them.",
    "memory": "Load user.md, memory.md, and session.md into the system prompt.",
    "skills": "Inject reusable workflow skills into prompts. (coming soon)",
    "curator": "Review and maintain skills and memory quality. (coming soon)",
    "kanban": "Track tasks on a local Kanban board. (coming soon)",
    "show_sources": "List retrieved document sources after each reply.",
    "streaming": "Stream model tokens into the chat as they are generated.",
}


class FeatureToggleModal(ModalScreen):
    """Modal for toggling runtime feature flags."""

    def __init__(self, current_state: dict[str, bool]) -> None:
        super().__init__()
        self.current_state = current_state

    def compose(self):
        with Vertical(id="feature-modal-container"):
            yield Label("Feature toggles (changes auto-save to config.yaml):")
            for key in FEATURE_KEYS:
                checkbox_id = f"feature-{key}"
                label = FEATURE_TOGGLE_LABELS.get(key, key)
                description = FEATURE_TOGGLE_DESCRIPTIONS.get(key, "")
                with Horizontal(classes="feature-row"):
                    yield Checkbox(label, id=checkbox_id, value=self.current_state.get(key, False))
                    yield Static(description, classes="feature-desc")
            with Container(id="button-container"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        for key in FEATURE_KEYS:
            checkbox = self.query_one(f"#feature-{key}", Checkbox)
            checkbox.value = self.current_state.get(key, False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok-button":
            selected = {
                key: self.query_one(f"#feature-{key}", Checkbox).value
                for key in FEATURE_KEYS
            }
            self.dismiss(selected)
        elif event.button.id == "cancel-button":
            self.dismiss(None)
