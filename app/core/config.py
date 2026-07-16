"""Configuration loading for SoulForge TUI.

All tunable values live in ``config.yaml`` at the project root. Core logic
should depend only on the typed dataclasses produced here, never on hardcoded
paths or magic numbers.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def resolve_path(value: str | Path) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ModelConfig:
    chat_model_path: str
    embedding_model_path: str
    context_size: int = 8192
    gpu_layers: int = -1
    threads: int = 8
    chat_format: str = "mistral-instruct"

    @property
    def chat_model(self) -> Path:
        return resolve_path(self.chat_model_path)

    @property
    def embedding_model(self) -> Path:
        return resolve_path(self.embedding_model_path)


@dataclass
class GenerationConfig:
    temperature: float = 0.75
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 700
    stop: list[str] = field(default_factory=lambda: ["</s>", "User:", "You:"])


@dataclass
class FeatureConfig:
    soul: bool = True
    rag: bool = True
    memory: bool = True
    skills: bool = False
    curator: bool = False
    kanban: bool = False
    agents: bool = False
    streaming: bool = True
    show_sources: bool = True
    tools: bool = False


# Maps FeatureConfig attribute names to config.yaml camelCase keys.
FEATURE_YAML_KEYS: dict[str, str] = {
    "soul": "soul",
    "rag": "rag",
    "memory": "memory",
    "skills": "skills",
    "curator": "curator",
    "kanban": "kanban",
    "agents": "agents",
    "streaming": "streaming",
    "show_sources": "showSources",
    "tools": "tools",
}

# Short labels used in the status bar and /features list.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "soul": "soul",
    "rag": "rag",
    "memory": "memory",
    "skills": "skills",
    "curator": "curator",
    "kanban": "kanban",
    "agents": "agents",
    "show_sources": "sources",
    "streaming": "streaming",
    "tools": "tools",
}


@dataclass
class RagConfig:
    db_path: str = "./chromaDb"
    docs_path: str = "./docs"
    collection_name: str = "localDocs"
    top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    pdf_ocr_enabled: bool = True
    pdf_ocr_lang: str = "eng"
    pdf_min_text_chars: int = 32

    @property
    def db_dir(self) -> Path:
        return resolve_path(self.db_path)

    @property
    def docs_dir(self) -> Path:
        return resolve_path(self.docs_path)


@dataclass
class MemoryConfig:
    user_file: str = "./app/memory/user.md"
    memory_file: str = "./app/memory/memory.md"
    session_file: str = "./app/memory/session.md"
    update_every_turns: int = 10
    max_user_chars: int = 3000
    max_memory_chars: int = 6000
    max_session_chars: int = 4000

    @property
    def user_path(self) -> Path:
        return resolve_path(self.user_file)

    @property
    def memory_path(self) -> Path:
        return resolve_path(self.memory_file)

    @property
    def session_path(self) -> Path:
        return resolve_path(self.session_file)


@dataclass
class SkillsConfig:
    active_path: str = "./app/skills/active"
    archived_path: str = "./app/skills/archived"
    registry_path: str = "./app/skills/registry.json"
    workflow_log_path: str = "./app/skills/workflow_log.json"
    auto_create: bool = False
    min_successful_repeats: int = 3
    success_window_turns: int = 3

    @property
    def active_dir(self) -> Path:
        return resolve_path(self.active_path)

    @property
    def archived_dir(self) -> Path:
        return resolve_path(self.archived_path)

    @property
    def registry_file(self) -> Path:
        return resolve_path(self.registry_path)

    @property
    def workflow_log_file(self) -> Path:
        return resolve_path(self.workflow_log_path)


@dataclass
class CuratorConfig:
    stale_days: int = 30
    bloat_max_chars: int = 1000


@dataclass
class TasksConfig:
    kanban_path: str = "./app/tasks/kanban.json"

    @property
    def kanban_file(self) -> Path:
        return resolve_path(self.kanban_path)


@dataclass
class SessionsConfig:
    store_path: str = "./app/sessions"
    max_saved_sessions: int = 50

    @property
    def store_dir(self) -> Path:
        return resolve_path(self.store_path)


@dataclass
class AgentModelProfileConfig:
    chat_model_path: str | None = None
    residency: str = "swap"  # resident | swap
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None
    chat_format: str | None = None

    @property
    def chat_model(self) -> Path | None:
        if not self.chat_model_path:
            return None
        return resolve_path(self.chat_model_path)


@dataclass
class AgentRoleConfig:
    model_profile: str = "default"
    allowed_tools: list[str] = field(default_factory=list)


def default_agent_model_profiles() -> dict[str, AgentModelProfileConfig]:
    return {
        "orchestrator": AgentModelProfileConfig(
            residency="swap",
            temperature=0.2,
            max_tokens=1400,
        ),
        "creator": AgentModelProfileConfig(
            residency="resident",
            temperature=0.35,
            max_tokens=1800,
        ),
        "critic_executor": AgentModelProfileConfig(
            residency="resident",
            temperature=0.1,
            max_tokens=1000,
        ),
    }


def default_agent_roles() -> dict[str, AgentRoleConfig]:
    return {
        "orchestrator": AgentRoleConfig(model_profile="orchestrator"),
        "researcher": AgentRoleConfig(
            model_profile="critic_executor",
            allowed_tools=["read_file", "list_dir", "search_docs"],
        ),
        "creator": AgentRoleConfig(model_profile="creator"),
        "executor": AgentRoleConfig(
            model_profile="critic_executor",
            allowed_tools=["read_file", "list_dir", "run_command", "write_file"],
        ),
        "critic": AgentRoleConfig(
            model_profile="critic_executor",
            allowed_tools=["read_file", "list_dir", "search_docs"],
        ),
        "synthesizer": AgentRoleConfig(model_profile="creator"),
    }


@dataclass
class AgentsConfig:
    runs_path: str = "./app/agents/runs"
    max_iterations: int = 3
    require_approval: bool = True
    strict_json: bool = True
    residency_mode: str = "hybrid"  # hybrid | sequential
    default_profile: str = "creator"
    model_profiles: dict[str, AgentModelProfileConfig] = field(
        default_factory=default_agent_model_profiles
    )
    roles: dict[str, AgentRoleConfig] = field(default_factory=default_agent_roles)

    @property
    def runs_dir(self) -> Path:
        return resolve_path(self.runs_path)


@dataclass
class LoggingConfig:
    log_path: str = "./logs/soulforge.log"
    level: str = "info"
    console: bool = False


@dataclass
class ToolsConfig:
    allow_write: bool = False
    allow_shell: bool = False
    read_roots: list[str] = field(
        default_factory=lambda: ["./docs", "./app"]
    )
    write_roots: list[str] = field(
        default_factory=lambda: ["./app/memory", "./app/tasks"]
    )
    max_read_bytes: int = 65536
    shell_allowlist: list[str] = field(default_factory=list)
    allow_network: bool = False
    network_allowlist: list[str] = field(default_factory=list)
    auto_approve_read_only: bool = True

    @property
    def read_root_paths(self) -> list[Path]:
        return [resolve_path(value) for value in self.read_roots]

    @property
    def write_root_paths(self) -> list[Path]:
        return [resolve_path(value) for value in self.write_roots]


@dataclass
class OnboardingConfig:
    completed: bool = False


@dataclass
class ServerConfig:
    """Local API server used by the desktop GUI front end."""

    host: str = "127.0.0.1"
    port: int = 8765
    # Shared secret sent as the ``X-SoulForge-Token`` header. Empty disables the
    # check (fine for single-user localhost). WSL2 forwards localhost to the
    # Windows host, so keep the bind on the loopback interface.
    auth_token: str = ""


@dataclass
class VisionConfig:
    """Optional multimodal model for the GUI's screen-snapshot feature.

    Requires a vision GGUF plus its mmproj (CLIP projector) file. Loaded on
    demand by ``ModelRuntime``; empty ``model_path`` disables the feature (the
    snapshot endpoint then falls back to OCR).
    """

    model_path: str = ""
    mmproj_path: str = ""
    chat_handler: str = "llava-1-5"  # llava-1-5 | llava-1-6 | moondream | qwen2.5-vl
    context_size: int = 4096
    max_tokens: int = 512
    # Free the chat model's VRAM before loading the vision model. Safer on tight
    # budgets (the chat model reloads on the next message); costs a reload.
    evict_chat: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.model_path)

    @property
    def model(self) -> Path | None:
        return resolve_path(self.model_path) if self.model_path else None

    @property
    def mmproj(self) -> Path | None:
        return resolve_path(self.mmproj_path) if self.mmproj_path else None


@dataclass
class TranscriptionConfig:
    """faster-whisper settings for the GUI's push-to-talk hotkey.

    Runs on the WSL GPU alongside the chat model. ``model_size`` accepts any
    faster-whisper size (tiny/base/small/medium/large-v3) or a local path.
    """

    model_size: str = "small"
    device: str = "cuda"  # cuda | cpu
    compute_type: str = "float16"  # float16 | int8_float16 | int8
    language: str = ""  # "" = autodetect


@dataclass
class AppConfig:
    model: ModelConfig
    generation: GenerationConfig
    features: FeatureConfig
    rag: RagConfig
    memory: MemoryConfig
    skills: SkillsConfig
    curator: CuratorConfig
    tasks: TasksConfig
    sessions: SessionsConfig
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    onboarding: OnboardingConfig = field(default_factory=OnboardingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.yaml`` into typed dataclasses."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy or create config.yaml at the project root."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_section = _section(data, "model")
    model = ModelConfig(
        chat_model_path=model_section.get(
            "chatModelPath", "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf"
        ),
        embedding_model_path=model_section.get(
            "embeddingModelPath", "./models/embedding-model.gguf"
        ),
        context_size=model_section.get("contextSize", 8192),
        gpu_layers=model_section.get("gpuLayers", -1),
        threads=model_section.get("threads", 8),
        chat_format=model_section.get("chatFormat", "mistral-instruct"),
    )

    gen_section = _section(data, "generation")
    generation = GenerationConfig(
        temperature=gen_section.get("temperature", 0.75),
        top_p=gen_section.get("topP", 0.95),
        repeat_penalty=gen_section.get("repeatPenalty", 1.1),
        max_tokens=gen_section.get("maxTokens", 700),
        stop=gen_section.get("stop", ["</s>", "User:", "You:"]),
    )

    feat_section = _section(data, "features")
    features = FeatureConfig(
        soul=feat_section.get("soul", True),
        rag=feat_section.get("rag", True),
        memory=feat_section.get("memory", True),
        skills=feat_section.get("skills", False),
        curator=feat_section.get("curator", False),
        kanban=feat_section.get("kanban", False),
        agents=feat_section.get("agents", False),
        streaming=feat_section.get("streaming", True),
        show_sources=feat_section.get("showSources", True),
        tools=feat_section.get("tools", False),
    )

    rag_section = _section(data, "rag")
    rag = RagConfig(
        db_path=rag_section.get("dbPath", "./chromaDb"),
        docs_path=rag_section.get("docsPath", "./docs"),
        collection_name=rag_section.get("collectionName", "localDocs"),
        top_k=rag_section.get("topK", 5),
        chunk_size=rag_section.get("chunkSize", 1200),
        chunk_overlap=rag_section.get("chunkOverlap", 200),
        pdf_ocr_enabled=rag_section.get("pdfOcrEnabled", True),
        pdf_ocr_lang=rag_section.get("pdfOcrLang", "eng"),
        pdf_min_text_chars=rag_section.get("pdfMinTextChars", 32),
    )

    mem_section = _section(data, "memory")
    memory = MemoryConfig(
        user_file=mem_section.get("userFile", "./app/memory/user.md"),
        memory_file=mem_section.get("memoryFile", "./app/memory/memory.md"),
        session_file=mem_section.get("sessionFile", "./app/memory/session.md"),
        update_every_turns=mem_section.get("updateEveryTurns", 10),
        max_user_chars=mem_section.get("maxUserChars", 3000),
        max_memory_chars=mem_section.get("maxMemoryChars", 6000),
        max_session_chars=mem_section.get("maxSessionChars", 4000),
    )

    skills_section = _section(data, "skills")
    skills = SkillsConfig(
        active_path=skills_section.get("activePath", "./app/skills/active"),
        archived_path=skills_section.get("archivedPath", "./app/skills/archived"),
        registry_path=skills_section.get("registryPath", "./app/skills/registry.json"),
        workflow_log_path=skills_section.get(
            "workflowLogPath", "./app/skills/workflow_log.json"
        ),
        auto_create=skills_section.get("autoCreate", False),
        min_successful_repeats=skills_section.get("minSuccessfulRepeats", 3),
        success_window_turns=skills_section.get("successWindowTurns", 3),
    )

    curator_section = _section(data, "curator")
    curator = CuratorConfig(
        stale_days=curator_section.get("staleDays", 30),
        bloat_max_chars=curator_section.get("bloatMaxChars", 1000),
    )

    tasks_section = _section(data, "tasks")
    tasks = TasksConfig(
        kanban_path=tasks_section.get("kanbanPath", "./app/tasks/kanban.json"),
    )

    sessions_section = _section(data, "sessions")
    sessions = SessionsConfig(
        store_path=sessions_section.get("storePath", "./app/sessions"),
        max_saved_sessions=sessions_section.get("maxSavedSessions", 50),
    )

    agents_section = _section(data, "agents")
    profiles = default_agent_model_profiles()
    raw_profiles = agents_section.get("modelProfiles", {})
    if isinstance(raw_profiles, dict):
        for name, profile_data in raw_profiles.items():
            if not isinstance(profile_data, dict):
                continue
            profiles[str(name)] = AgentModelProfileConfig(
                chat_model_path=profile_data.get("chatModelPath"),
                residency=profile_data.get("residency", "swap"),
                temperature=profile_data.get("temperature"),
                top_p=profile_data.get("topP"),
                repeat_penalty=profile_data.get("repeatPenalty"),
                max_tokens=profile_data.get("maxTokens"),
                chat_format=profile_data.get("chatFormat"),
            )

    roles = default_agent_roles()
    raw_roles = agents_section.get("roles", {})
    if isinstance(raw_roles, dict):
        for name, role_data in raw_roles.items():
            if not isinstance(role_data, dict):
                continue
            tools = role_data.get("allowedTools", [])
            if not isinstance(tools, list):
                tools = []
            existing = roles.get(str(name), AgentRoleConfig())
            roles[str(name)] = AgentRoleConfig(
                model_profile=role_data.get(
                    "modelProfile",
                    existing.model_profile,
                ),
                allowed_tools=[str(tool) for tool in tools],
            )

    agents = AgentsConfig(
        runs_path=agents_section.get("runsPath", "./app/agents/runs"),
        max_iterations=agents_section.get("maxIterations", 3),
        require_approval=agents_section.get("requireApproval", True),
        strict_json=agents_section.get("strictJson", True),
        residency_mode=agents_section.get("residencyMode", "hybrid"),
        default_profile=agents_section.get("defaultProfile", "creator"),
        model_profiles=profiles,
        roles=roles,
    )

    logging_section = _section(data, "logging")
    logging_cfg = LoggingConfig(
        log_path=logging_section.get("logPath", "./logs/soulforge.log"),
        level=logging_section.get("level", "info"),
        console=logging_section.get("console", False),
    )

    tools_section = _section(data, "tools")
    tools_cfg = ToolsConfig(
        allow_write=tools_section.get("allowWrite", False),
        allow_shell=tools_section.get("allowShell", False),
        read_roots=tools_section.get("readRoots", ["./docs", "./app"]),
        write_roots=tools_section.get(
            "writeRoots", ["./app/memory", "./app/tasks"]
        ),
        max_read_bytes=tools_section.get("maxReadBytes", 65536),
        shell_allowlist=tools_section.get("shellAllowlist", []),
        allow_network=tools_section.get("allowNetwork", False),
        network_allowlist=tools_section.get("networkAllowlist", []),
        auto_approve_read_only=tools_section.get("autoApproveReadOnly", True),
    )

    onboarding_section = _section(data, "onboarding")
    onboarding = OnboardingConfig(
        completed=onboarding_section.get("completed", False),
    )

    server_section = _section(data, "server")
    server = ServerConfig(
        host=server_section.get("host", "127.0.0.1"),
        port=server_section.get("port", 8765),
        auth_token=server_section.get("authToken", ""),
    )

    vision_section = _section(data, "vision")
    vision = VisionConfig(
        model_path=vision_section.get("modelPath", ""),
        mmproj_path=vision_section.get("mmprojPath", ""),
        chat_handler=vision_section.get("chatHandler", "llava-1-5"),
        context_size=vision_section.get("contextSize", 4096),
        max_tokens=vision_section.get("maxTokens", 512),
        evict_chat=vision_section.get("evictChat", False),
    )

    transcription_section = _section(data, "transcription")
    transcription = TranscriptionConfig(
        model_size=transcription_section.get("modelSize", "small"),
        device=transcription_section.get("device", "cuda"),
        compute_type=transcription_section.get("computeType", "float16"),
        language=transcription_section.get("language", ""),
    )

    return AppConfig(
        model=model,
        generation=generation,
        features=features,
        rag=rag,
        memory=memory,
        skills=skills,
        curator=curator,
        tasks=tasks,
        sessions=sessions,
        agents=agents,
        logging=logging_cfg,
        tools=tools_cfg,
        onboarding=onboarding,
        server=server,
        vision=vision,
        transcription=transcription,
        raw=data,
    )


def features_to_yaml_dict(features: FeatureConfig) -> dict[str, bool]:
    """Convert a FeatureConfig to the camelCase dict written under ``features:``."""
    return {
        yaml_key: getattr(features, attr)
        for attr, yaml_key in FEATURE_YAML_KEYS.items()
    }


def tools_to_yaml_dict(tools: ToolsConfig) -> dict[str, Any]:
    """Convert a ToolsConfig to the camelCase dict written under ``tools:``."""
    return {
        "allowWrite": tools.allow_write,
        "allowShell": tools.allow_shell,
        "readRoots": list(tools.read_roots),
        "writeRoots": list(tools.write_roots),
        "maxReadBytes": tools.max_read_bytes,
        "shellAllowlist": list(tools.shell_allowlist),
        "allowNetwork": tools.allow_network,
        "networkAllowlist": list(tools.network_allowlist),
        "autoApproveReadOnly": tools.auto_approve_read_only,
    }


def agent_model_profile_to_yaml_dict(
    profile: AgentModelProfileConfig,
) -> dict[str, Any]:
    """Convert an agent model profile to the dict written under ``agents:``."""
    data: dict[str, Any] = {
        "chatModelPath": profile.chat_model_path,
        "residency": profile.residency,
    }
    if profile.temperature is not None:
        data["temperature"] = profile.temperature
    if profile.top_p is not None:
        data["topP"] = profile.top_p
    if profile.repeat_penalty is not None:
        data["repeatPenalty"] = profile.repeat_penalty
    if profile.max_tokens is not None:
        data["maxTokens"] = profile.max_tokens
    if profile.chat_format is not None:
        data["chatFormat"] = profile.chat_format
    return data


def agent_role_to_yaml_dict(role: AgentRoleConfig) -> dict[str, Any]:
    """Convert an agent role mapping to the dict written under ``agents:``."""
    data: dict[str, Any] = {"modelProfile": role.model_profile}
    if role.allowed_tools:
        data["allowedTools"] = list(role.allowed_tools)
    return data


def agents_to_yaml_dict(agents: AgentsConfig) -> dict[str, Any]:
    """Convert AgentsConfig to the camelCase dict written under ``agents:``."""
    return {
        "runsPath": agents.runs_path,
        "maxIterations": agents.max_iterations,
        "requireApproval": agents.require_approval,
        "strictJson": agents.strict_json,
        "residencyMode": agents.residency_mode,
        "defaultProfile": agents.default_profile,
        "modelProfiles": {
            name: agent_model_profile_to_yaml_dict(profile)
            for name, profile in agents.model_profiles.items()
        },
        "roles": {
            name: agent_role_to_yaml_dict(role)
            for name, role in agents.roles.items()
        },
    }


def save_agents(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current agents section to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    data["agents"] = agents_to_yaml_dict(config.agents)
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def save_tools(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current tools section to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    data["tools"] = tools_to_yaml_dict(config.tools)
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def save_features(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current feature flags to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    data["features"] = features_to_yaml_dict(config.features)
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def onboarding_to_yaml_dict(onboarding: OnboardingConfig) -> dict[str, bool]:
    """Convert an OnboardingConfig to the dict written under ``onboarding:``."""
    return {"completed": onboarding.completed}


def model_to_yaml_dict(model: ModelConfig) -> dict[str, Any]:
    """Convert a ModelConfig to the camelCase dict written under ``model:``."""
    return {
        "chatModelPath": model.chat_model_path,
        "embeddingModelPath": model.embedding_model_path,
        "contextSize": model.context_size,
        "gpuLayers": model.gpu_layers,
        "threads": model.threads,
        "chatFormat": model.chat_format,
    }


def save_chat_model(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current chat model path to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    model_section = dict(data.get("model") or {})
    model_section["chatModelPath"] = config.model.chat_model_path
    data["model"] = model_section
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def save_onboarding(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current onboarding state to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    data["onboarding"] = onboarding_to_yaml_dict(config.onboarding)
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
