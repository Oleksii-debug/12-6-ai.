"""Project-owned, fail-closed Playwright browser-runtime boundary.

This module contains only the project contract. The real Playwright dependency is
loaded lazily and only after exact-version verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from time import perf_counter
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

UPSTREAM_REPOSITORY = "https://github.com/microsoft/playwright"
UPSTREAM_TAG = "v1.62.0"
UPSTREAM_COMMIT = "e3950d9c140d007bd52853b45813c6274b24e36f"
UPSTREAM_LICENSE = "Apache-2.0"
UPSTREAM_LICENSE_BLOB_SHA = "df112373eb2e23e459bf93ec412be1764dc5a38b"
PACKAGE_NAME = "playwright"
PACKAGE_VERSION = "1.62.0"
PACKAGE_WHEEL_SHA256 = "ba33bae6a13b3d9d354c751cb618af357d20fe1d57767cbcce52079bbef17ad3"
PYYYEE_VERSION = "13.0.1"
GREENLET_VERSION = "3.2.3"
TYPING_EXTENSIONS_VERSION = "4.16.0"
PYTHON_REQUIREMENT = ">=3.10"
BROWSER_NAME = "chromium"
BROWSER_REVISION = "1234"

ALLOWED_ACTIONS = frozenset(
    {"open_page", "read_page", "click_element", "type_text", "select_option", "wait_for_element"}
)
SENSITIVE_ENV_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "AUTH")


class ContractError(ValueError):
    """Raised when the project browser contract is violated."""


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def validate_action(action: str) -> None:
    if action not in ALLOWED_ACTIONS:
        raise ContractError(f"unsupported browser action: {action}")


def validate_env_allowlist(env: Mapping[str, str]) -> None:
    for name in env:
        upper = name.upper()
        if any(word in upper for word in SENSITIVE_ENV_WORDS):
            raise ContractError(
                f"implicit credential-bearing environment variable rejected: {name}"
            )


def validate_url(
    url: str, *, allowed_hosts: frozenset[str] = frozenset(), network_mode: str = "deny"
) -> None:
    url = _require_string(url, "url")
    parsed = urlparse(url)
    if parsed.scheme in {"about", "data"} and network_mode == "deny" and not parsed.netloc:
        return
    if parsed.scheme not in {"http", "https"}:
        raise ContractError("only http/https URLs are permitted when network access is requested")
    if network_mode != "allowlist":
        raise ContractError("network access requires explicit allowlist mode")
    host = (parsed.hostname or "").lower()
    if not host or host not in allowed_hosts:
        raise ContractError(f"host is not in explicit network allowlist: {host or '<missing>'}")


def validate_selector(selector: str) -> None:
    selector = _require_string(selector, "selector")
    if selector.startswith("xpath=") or selector.startswith("coordinates:"):
        raise ContractError("coordinate/XPath targeting is outside the bounded semantic surface")


@dataclass(frozen=True)
class RuntimeContract:
    """Immutable runtime policy supplied by the trusted host."""

    allowed_hosts: frozenset[str] = frozenset()
    network_mode: str = "deny"
    env_allowlist: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.network_mode not in {"deny", "allowlist"}:
            raise ContractError("network_mode must be deny or allowlist")
        validate_env_allowlist({name: "<redacted>" for name in self.env_allowlist})
        for host in self.allowed_hosts:
            if not isinstance(host, str) or not host or host != host.lower():
                raise ContractError("allowed_hosts must contain lowercase non-empty hostnames")


@dataclass(frozen=True)
class RuntimeIdentity:
    python_playwright_version: str
    chromium_revision: str


def installed_identity() -> RuntimeIdentity:
    """Read the real installed package closure and require the exact pinned versions."""
    expected = {
        PACKAGE_NAME: PACKAGE_VERSION,
        "pyee": PYYYEE_VERSION,
        "greenlet": GREENLET_VERSION,
        "typing-extensions": TYPING_EXTENSIONS_VERSION,
    }
    for name, wanted in expected.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise ContractError(f"exact dependency is not installed: {name}=={wanted}") from exc
        if version != wanted:
            raise ContractError(f"exact dependency {name}=={wanted} required; found {version}")
    return RuntimeIdentity(PACKAGE_VERSION, BROWSER_REVISION)


def require_exact_runtime() -> Any:
    """Import the real dependency only after exact version verification."""
    installed_identity()
    try:
        import playwright.sync_api as sync_api  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ContractError(
            "Playwright package metadata is present but runtime import failed"
        ) from exc
    return sync_api


class PlaywrightBrowserSession:
    """Bounded semantic browser adapter around a real Playwright page."""

    def __init__(self, contract: RuntimeContract) -> None:
        contract.validate()
        self.contract = contract
        self._sync_api = require_exact_runtime()
        self._pw = None
        self._browser = None
        self._page = None

    def __enter__(self) -> "PlaywrightBrowserSession":
        self._pw = self._sync_api.sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    def _page_or_error(self) -> Any:
        if self._page is None:
            raise ContractError("browser session is not started")
        return self._page

    def open_page(self, url: str) -> None:
        validate_action("open_page")
        validate_url(
            url,
            allowed_hosts=self.contract.allowed_hosts,
            network_mode=self.contract.network_mode,
        )
        self._page_or_error().goto(url)

    def read_page(self) -> str:
        validate_action("read_page")
        return self._page_or_error().locator("body").inner_text()

    def click_element(self, *, role: str, name: str) -> None:
        validate_action("click_element")
        _require_string(role, "role")
        _require_string(name, "name")
        self._page_or_error().get_by_role(role, name=name).click()

    def type_text(self, *, label: str, text: str) -> None:
        validate_action("type_text")
        _require_string(label, "label")
        if not isinstance(text, str):
            raise ContractError("text must be a string")
        self._page_or_error().get_by_label(label).fill(text)

    def select_option(self, *, label: str, value: str) -> None:
        validate_action("select_option")
        _require_string(label, "label")
        _require_string(value, "value")
        self._page_or_error().get_by_label(label).select_option(value)

    def wait_for_element(self, *, role: str, name: str) -> None:
        validate_action("wait_for_element")
        _require_string(role, "role")
        _require_string(name, "name")
        self._page_or_error().get_by_role(role, name=name).wait_for(state="visible")


def run_real_smoke(
    contract: RuntimeContract, *, timer: Callable[[], float] = perf_counter
) -> dict[str, Any]:
    """Run a bounded real Playwright smoke on a data: page only."""
    contract.validate()
    if contract.network_mode != "deny":
        raise ContractError("V1 smoke benchmark is local/data-only and requires network_mode=deny")
    started = timer()
    with PlaywrightBrowserSession(contract) as browser:
        smoke_url = (
            "data:text/html,<main>"
            "<label for='i'>Name</label><input id='i'>"
            "<label for='s'>Choice</label>"
            "<select id='s'><option value='a'>A</option><option value='b'>B</option></select>"
            "<button>Run</button></main>"
        )
        browser.open_page(smoke_url)
        browser.wait_for_element(role="button", name="Run")
        browser.click_element(role="button", name="Run")
        browser.type_text(label="Name", text="ok")
        browser.select_option(label="Choice", value="b")
        body = browser.read_page()
    elapsed_ms = (timer() - started) * 1000.0
    return {
        "runtime": "EXECUTED_PASS",
        "action_count": 6,
        "body_text": body,
        "elapsed_ms": round(elapsed_ms, 3),
    }
