from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pytest

from twelve_six.inference import cli


@dataclass
class _PartialTextStream:
    text: str
    max_chunk: int
    position: int = 0
    read_sizes: list[int] = field(default_factory=list)

    def isatty(self) -> bool:
        return False

    def read(self, size: int) -> str:
        if not isinstance(size, int) or size <= 0:
            raise AssertionError("bounded stdin reader must always request a positive size")
        self.read_sizes.append(size)
        if self.position >= len(self.text):
            return ""
        take = min(size, self.max_chunk, len(self.text) - self.position)
        chunk = self.text[self.position : self.position + take]
        self.position += take
        return chunk


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="prompt-bound-test")


def test_stdin_exact_boundary_is_returned_without_truncation(monkeypatch) -> None:
    stream = _PartialTextStream("абвгд", max_chunk=2)
    monkeypatch.setattr(cli.sys, "stdin", stream)

    prompt = cli._read_prompt(_parser(), None, max_prompt_chars=5)

    assert prompt == "абвгд"
    assert len(prompt) == 5
    assert len(stream.read_sizes) >= 3
    assert all(0 < size <= cli.PROMPT_READ_CHUNK_CHARS for size in stream.read_sizes)


def test_stdin_overflow_is_detected_even_when_stream_returns_short_chunks(
    monkeypatch,
    capsys,
) -> None:
    secret = "DO-NOT-ECHO-THIS-PROMPT"
    stream = _PartialTextStream(secret, max_chunk=3)
    monkeypatch.setattr(cli.sys, "stdin", stream)

    with pytest.raises(SystemExit) as exc_info:
        cli._read_prompt(_parser(), None, max_prompt_chars=7)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "stdin prompt exceeds --max-prompt-chars" in captured.err
    assert secret not in captured.err
    assert sum(min(size, stream.max_chunk) for size in stream.read_sizes) >= 8
    assert all(size > 0 for size in stream.read_sizes)


def test_bounded_stdin_never_requests_more_than_limit_plus_sentinel(monkeypatch) -> None:
    stream = _PartialTextStream("x" * 100, max_chunk=100)
    monkeypatch.setattr(cli.sys, "stdin", stream)

    value = cli._read_bounded_stdin(9)

    assert value == "x" * 10
    assert stream.read_sizes == [10]
    assert stream.position == 10


def test_direct_prompt_uses_same_limit_and_does_not_echo_content(capsys) -> None:
    secret = "PRIVATE-PROMPT-CONTENT"

    with pytest.raises(SystemExit) as exc_info:
        cli._read_prompt(_parser(), secret, max_prompt_chars=5)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--prompt exceeds --max-prompt-chars" in captured.err
    assert secret not in captured.err


def test_direct_prompt_at_limit_is_accepted() -> None:
    assert cli._read_prompt(_parser(), "12345", max_prompt_chars=5) == "12345"


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_prompt_limit_is_rejected_by_argparse(value: int) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--prompt",
                "x",
                "--max-prompt-chars",
                str(value),
            ]
        )

    assert exc_info.value.code == 2


def test_prompt_limit_default_is_explicit_and_positive() -> None:
    args = cli.build_parser().parse_args(["--checkpoint", "checkpoint", "--prompt", "x"])

    assert args.max_prompt_chars == cli.DEFAULT_MAX_PROMPT_CHARS
    assert args.max_prompt_chars > cli.PROMPT_READ_CHUNK_CHARS


@pytest.mark.parametrize("value", [True, 0, -1])
def test_internal_prompt_limit_contract_fails_closed(value: object) -> None:
    expected = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(expected, match="positive integer"):
        cli._read_prompt(_parser(), "x", max_prompt_chars=value)  # type: ignore[arg-type]
