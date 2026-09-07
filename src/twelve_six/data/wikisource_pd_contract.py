"""Pinned authority and normalization contract for the Lesia 1892 Wikisource edition."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

API_URL = "https://uk.wikisource.org/w/api.php"
SOURCE_FAMILY_ID = "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv"
INDEX_REVISION_ID = 729499
INCUMBENT_HEAD_SHA = "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2"
INCUMBENT_AUTHORITY_SHA256 = (
    "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6"
)
PAGE_PREFIX = "Сторінка:Леся Українка. На крилах пісень. 1892.pdf/"
APPROVED_CATEGORY = "Перевірені сторінки"


class WikisourceIntakeError(ValueError):
    """Raised when a source or materialization boundary fails closed."""


def validate_control_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != "12-6.d03-wikisource-lesia1892-current-main.v1":
        raise WikisourceIntakeError("unexpected Wikisource control schema")
    if contract.get("execution_class") != "LOCAL_FREE":
        raise WikisourceIntakeError("only LOCAL_FREE execution is authorized")
    incumbent = contract.get("incumbent_authority")
    if not isinstance(incumbent, dict):
        raise WikisourceIntakeError("incumbent authority is missing")
    if incumbent.get("head_sha") != INCUMBENT_HEAD_SHA:
        raise WikisourceIntakeError("incumbent exact head drift")
    if incumbent.get("authority_identity_sha256") != INCUMBENT_AUTHORITY_SHA256:
        raise WikisourceIntakeError("incumbent authority identity drift")
    edition = contract.get("edition")
    if not isinstance(edition, dict):
        raise WikisourceIntakeError("edition contract is missing")
    if edition.get("index_revision_id") != INDEX_REVISION_ID:
        raise WikisourceIntakeError("edition index revision drift")
    if edition.get("source_family_id") != SOURCE_FAMILY_ID:
        raise WikisourceIntakeError("source family drift")
    if edition.get("family_credit_added") is not False:
        raise WikisourceIntakeError("same-edition expansion may not add family credit")
    acquisition = contract.get("acquisition")
    if not isinstance(acquisition, dict) or acquisition.get("max_pages") != 112:
        raise WikisourceIntakeError("bounded acquisition contract drift")
    if acquisition.get("minimum_request_cadence_seconds", 0) < 0.5:
        raise WikisourceIntakeError("request cadence weakened")
    boundary = contract.get("truth_boundary")
    if not isinstance(boundary, dict):
        raise WikisourceIntakeError("truth boundary is missing")
    zero_fields = (
        "canonical_capacity_credit_bytes",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "optimizer_updates",
    )
    if any(boundary.get(field) != 0 for field in zero_fields):
        raise WikisourceIntakeError("candidate-only zero-credit boundary was promoted")
    false_fields = (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_outcomes_read",
        "paid_compute_used",
    )
    if any(boundary.get(field) is not False for field in false_fields):
        raise WikisourceIntakeError("candidate-only execution boundary was promoted")


def normalize_rendered_text(text: str) -> str:
    """Normalize source-rendered literary text without modernizing its contents."""
    if not isinstance(text, str):
        raise WikisourceIntakeError("rendered text must be a string")
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if line.startswith("\u00a0"):
            if lines and lines[-1] != "":
                lines.append("")
            line = line.lstrip("\u00a0 ")
        lines.append(line)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    compact: list[str] = []
    for line in lines:
        if line or not compact or compact[-1]:
            compact.append(line)
    normalized = "\n".join(compact).strip() + "\n"
    if "\u00a0" in normalized or "\r" in normalized:
        raise WikisourceIntakeError("normalization left forbidden line markers")
    return normalized


def validate_ua_page_text(text: str) -> None:
    if len(text.encode("utf-8")) < 64:
        raise WikisourceIntakeError("page body is too short")
    lower = text.lower()
    if any(marker in lower for marker in ("<html", "mw-parser-output", "перегляд історії")):
        raise WikisourceIntakeError("page body contains site chrome or raw HTML")
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        raise WikisourceIntakeError("page body has no alphabetic content")
    ua_alphabet = set("іїєґІЇЄҐабвгдежзиклмнопрстуфхцчшщьюяАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЬЮЯ")
    if sum(ch in ua_alphabet for ch in letters) / len(letters) < 0.75:
        raise WikisourceIntakeError("page body is not predominantly Ukrainian Cyrillic")
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
        raise WikisourceIntakeError("email-like content requires downstream privacy quarantine")


def validate_page_title(title: str) -> int:
    if not title.startswith(PAGE_PREFIX):
        raise WikisourceIntakeError("page title is outside the qualified edition")
    suffix = title.removeprefix(PAGE_PREFIX)
    if not suffix.isdigit():
        raise WikisourceIntakeError("only numeric scan pages are eligible")
    page_number = int(suffix)
    if not 1 <= page_number <= 112:
        raise WikisourceIntakeError("scan page is outside the pinned edition bounds")
    return page_number
