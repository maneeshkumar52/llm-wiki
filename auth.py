"""Local authentication helpers for workspace access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils import sha256_hex


@dataclass(slots=True)
class AuthUser:
    """Authenticated user configuration."""

    username: str
    display_name: str
    role: str
    workspaces: list[str]


class LocalAuthManager:
    """Simple local auth layer backed by config.yaml."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.security = config.get("security", {})
        self.workspaces = config.get("workspaces", {})

    @property
    def enabled(self) -> bool:
        """Return whether login is required."""
        return bool(self.security.get("enabled", False))

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        """Validate a username/password pair against configured users."""
        for entry in self.security.get("users", []):
            configured_username = str(entry.get("username", ""))
            if configured_username != username:
                continue
            password_hash = str(entry.get("password_sha256", ""))
            if sha256_hex(password) != password_hash:
                return None
            return AuthUser(
                username=configured_username,
                display_name=str(entry.get("display_name", configured_username)),
                role=str(entry.get("role", "member")),
                workspaces=[str(item) for item in entry.get("workspaces", [])],
            )
        return None

    def get_default_user(self) -> AuthUser:
        """Return the implicit local user when auth is disabled."""
        defaults = [str(item) for item in self.workspaces.get("defaults", ["General"])]
        return AuthUser(
            username="local-admin",
            display_name="Local Admin",
            role="owner",
            workspaces=defaults,
        )

    def get_workspace_options(self, user: AuthUser) -> list[str]:
        """Return workspaces visible to the given user."""
        defaults = [str(item) for item in self.workspaces.get("defaults", ["General"])]
        options = user.workspaces or defaults
        return options or ["General"]