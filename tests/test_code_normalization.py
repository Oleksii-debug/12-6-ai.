from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import pytest

from twelve_six.data import pipeline
from twelve_six.data.code_normalization import (
    CODE_NORMALIZATION_POLICY,
    CodeNormalizationError,
    decode_code_bytes,
    extract_markdown_fenced_code,
    normalize_code_text,
)
from twelve_six.data.multilingual_pretraining import strict_normalize_utf8


FIDELITY_FIXTURES = {
    "python": (
        "# коментар: preserve Unicode and spacing\r\n"
        "café = \"① ﬁ K\"\r\n"
        "if café:\r\n"
        "\tvalue = café + \\\r\n"
        "\t    \"!\"  # trailing comment\r\n"
    ),
    "javascript": (
        "// preserve comments and Unicode identifiers\n"
        "const café = \"① ﬁ K\";\n"
        "function keep(value) {\n"
        "\tconst joined = \"left\\\nright\";\n"
        "\treturn `${café}:${value}:${joined}`;\n"
        "}\n"
    ),
    "typescript": (
        "// TypeScript fixture\n"
        "const café: string = \"① ﬁ K\";\n"
        "export function keep(value: number): string {\n"
        "\treturn `${café}:${value}`;\n"
        "}\n"
    ),
    "c": (
        "/* C comment */\n"
        "#define ADD(a, b) ((a) + \\\n"
        "                      (b))\n"
        "int main(void) {\n"
        "\tconst char *label = \"① ﬁ K\";\n"
        "\treturn label[0] == 0;\n"
        "}\n"
    ),
    "cpp": (
        "// C++ comment\n"
        "#include <string>\n"
        "int main() {\n"
        "\tstd::string label = \"① ﬁ K\";\n"
        "\treturn label.empty() ? 1 : 0;\n"
        "}\n"
    ),
    "json": (
        "{\n"
        "\t\"name\": \"① ﬁ K\",\n"
        "\t\"nested\": {\"keep\": true, \"count\": 2}\n"
        "}\n"
    ),
    "yaml": (
        "# YAML comment\n"
        "name: \"① ﬁ K\"\n"
        "items:\n"
        "  - one\n"
        "  - two\n"
        "literal: |\n"
        "  preserve  two  spaces\n"
    ),
    "shell": (
        "#!/bin/sh\n"
        "# shell comment\n"
        "name='① ﬁ K'\n"
        "printf '%s\\n' \\\n"
        "\t\"$name\"\n"
    ),
    "markdown": (
        "# Fidelity\r\n"
        "\r\n"
        "```python\r\n"
        "\tvalue = \"① ﬁ K\"  # fenced comment\r\n"
        "```\r\n"
        "\r\n"
        "~~~sh\r\n"
        "printf '%s\\n' \\\r\n"
        "\t\"ok\"\r\n"
        "~~~\r\n"
    ),
}


def _run_parser(command: list[str], text: str, suffix: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return subprocess.run(
            [part.replace("{path}", str(path)) for part in command],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("language,text", sorted(FIDELITY_FIXTURES.items()))
def test_code_normalization_is_exact_identity(language: str, text: str) -> None:
    normalized, evidence = normalize_code_text(text, language=language, path=f"sample.{language}")
    assert normalized == text
    assert normalized.encode("utf-8") == text.encode("utf-8")
    assert evidence.policy == CODE_NORMALIZATION_POLICY
    assert evidence.transformations == ()
    assert evidence.source_sha256 == evidence.normalized_sha256
    assert evidence.source_utf8_bytes == evidence.normalized_utf8_bytes
    assert evidence.structure.tabs == text.count("\t")
    assert evidence.structure.line_continuations == len(
        __import__("re").findall(r"\\(?:\r\n|\n|\r)", text)
    )


def test_regression_old_nfkc_would_mutate_code_but_identity_policy_does_not() -> None:
    source = 'café = "① ﬁ K"\r\n\t# keep exact source\r\n'
    old_data10_behavior = unicodedata.normalize(
        "NFKC", source.replace("\r\n", "\n").replace("\r", "\n")
    ).strip("\n")
    assert old_data10_behavior != source
    normalized, evidence = normalize_code_text(source, language="python", path="fixture.py")
    assert normalized == source
    assert evidence.source_sha256 == hashlib.sha256(source.encode()).hexdigest()


def test_multilingual_code_path_preserves_crlf_tabs_and_compatibility_characters() -> None:
    source = 'def café():\r\n\treturn "① ﬁ K"\r\n'
    normalized, _profile = strict_normalize_utf8(source, preserve_layout=True)
    assert normalized == source


def test_markdown_fenced_code_bodies_preserve_exact_layout() -> None:
    source = FIDELITY_FIXTURES["markdown"]
    normalized, _ = normalize_code_text(source, language="markdown", path="README.md")
    blocks = extract_markdown_fenced_code(normalized)
    assert len(blocks) == 2
    assert blocks[0].info == "python"
    assert blocks[0].body == '\tvalue = "① ﬁ K"  # fenced comment\r\n'
    assert blocks[1].info == "sh"
    assert blocks[1].body == 'printf \'%s\\n\' \\\r\n\t"ok"\r\n'


def test_python_and_json_parse_before_and_after() -> None:
    python_source = FIDELITY_FIXTURES["python"]
    python_normalized, _ = normalize_code_text(python_source, language="python")
    assert ast.dump(ast.parse(python_source)) == ast.dump(ast.parse(python_normalized))

    json_source = FIDELITY_FIXTURES["json"]
    json_normalized, _ = normalize_code_text(json_source, language="json")
    assert json.loads(json_source) == json.loads(json_normalized)


@pytest.mark.parametrize(
    "tool,language,suffix,command",
    [
        ("node", "javascript", ".js", ["node", "--check", "{path}"]),
        ("bash", "shell", ".sh", ["bash", "-n", "{path}"]),
    ],
)
def test_optional_script_parsers_before_and_after(
    tool: str, language: str, suffix: str, command: list[str]
) -> None:
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} parser is not installed")
    source = FIDELITY_FIXTURES[language]
    normalized, _ = normalize_code_text(source, language=language)
    before = _run_parser(command, source, suffix)
    after = _run_parser(command, normalized, suffix)
    assert before.returncode == 0, before.stderr
    assert after.returncode == 0, after.stderr


@pytest.mark.parametrize(
    "language,suffix,tools,arguments",
    [
        ("c", ".c", ("cc", "gcc", "clang"), ["-fsyntax-only", "{path}"]),
        ("cpp", ".cpp", ("c++", "g++", "clang++"), ["-fsyntax-only", "{path}"]),
    ],
)
def test_optional_c_family_parsers_before_and_after(
    language: str,
    suffix: str,
    tools: tuple[str, ...],
    arguments: list[str],
) -> None:
    compiler = next((tool for tool in tools if shutil.which(tool)), None)
    if compiler is None:
        pytest.skip(f"no {language} compiler is installed")
    source = FIDELITY_FIXTURES[language]
    normalized, _ = normalize_code_text(source, language=language)
    command = [compiler, *arguments]
    before = _run_parser(command, source, suffix)
    after = _run_parser(command, normalized, suffix)
    assert before.returncode == 0, before.stderr
    assert after.returncode == 0, after.stderr


def test_optional_yaml_parser_before_and_after() -> None:
    yaml = pytest.importorskip("yaml")
    source = FIDELITY_FIXTURES["yaml"]
    normalized, _ = normalize_code_text(source, language="yaml")
    assert yaml.safe_load(source) == yaml.safe_load(normalized)


def test_optional_typescript_parser_before_and_after() -> None:
    if shutil.which("tsc") is None:
        pytest.skip("tsc is not installed")
    source = FIDELITY_FIXTURES["typescript"]
    normalized, _ = normalize_code_text(source, language="typescript")
    before = _run_parser(["tsc", "--noEmit", "--pretty", "false", "{path}"], source, ".ts")
    after = _run_parser(
        ["tsc", "--noEmit", "--pretty", "false", "{path}"], normalized, ".ts"
    )
    assert before.returncode == 0, before.stdout + before.stderr
    assert after.returncode == 0, after.stdout + after.stderr


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b"\xff\xfe\x00bad", "binary_invalid_utf8"),
        (b"abc\x00def", "binary_nul"),
        (b"abc\x01def", "binary_control_character"),
    ],
)
def test_malformed_binary_material_is_rejected(payload: bytes, reason: str) -> None:
    with pytest.raises(CodeNormalizationError) as error:
        decode_code_bytes(payload, language="python", path="bad.py")
    assert error.value.reason == reason


def test_generated_and_minified_material_are_rejected_explicitly() -> None:
    generated = "// Code generated by fixture; DO NOT EDIT.\nconst x = 1;\n"
    with pytest.raises(CodeNormalizationError) as generated_error:
        normalize_code_text(generated, language="javascript", path="generated/output.js")
    assert generated_error.value.reason == "generated_material"

    with pytest.raises(CodeNormalizationError) as minified_error:
        normalize_code_text(
            "const x=1;const y=2;\n",
            language="javascript",
            path="bundle.min.js",
        )
    assert minified_error.value.reason == "minified_material"


def test_d03_manifest_records_code_raw_normalized_hashes_and_reason_counters(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / "data"
    raw_dir = registry_dir / "raw"
    raw_dir.mkdir(parents=True)
    records = [
        {
            "document_id": "py",
            "modality": "code",
            "code_language": "python",
            "path": "src/example.py",
            "text": FIDELITY_FIXTURES["python"],
        },
        {
            "document_id": "json",
            "modality": "code",
            "code_language": "json",
            "path": "config/example.json",
            "text": FIDELITY_FIXTURES["json"],
        },
        {
            "document_id": "shell",
            "modality": "code",
            "code_language": "shell",
            "path": "scripts/example.sh",
            "text": FIDELITY_FIXTURES["shell"],
        },
        {
            "document_id": "generated",
            "modality": "code",
            "code_language": "javascript",
            "path": "generated/output.js",
            "text": "// Code generated by fixture; DO NOT EDIT.\nconst x = 1;\n",
        },
    ]
    raw = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    (raw_dir / "code.jsonl").write_bytes(raw)
    registry = {
        "schema_version": 1,
        "dataset_id": "data28-code-fixture",
        "sources": [
            {
                "source_id": "project-code",
                "purpose": "pretraining",
                "raw_path": "raw/code.jsonl",
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "provenance": {
                    "synthetic": True,
                    "synthetic_kind": "data28_test_fixture",
                },
                "license": {"status": "PROJECT_AUTHORED"},
            }
        ],
    }
    registry_path = registry_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    contamination_path = registry_dir / "contamination.json"
    contamination_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "forbidden_normalized_sha256": [],
                "forbidden_source_purposes": ["benchmark", "evaluation_test", "heldout_test"],
            }
        ),
        encoding="utf-8",
    )

    manifest = pipeline.build_dataset(registry_path, contamination_path, tmp_path / "out")
    assert manifest["code_normalization_policy"]["transform"] == "identity"
    assert manifest["code_normalization_policy"]["natural_text_normalizer_applied_to_code"] is False
    assert manifest["stats"]["code_input_documents"] == 4
    assert manifest["stats"]["code_rejected_documents"] == 1
    assert manifest["stats"]["code_rejection_reasons"] == {"generated_material": 1}
    assert manifest["stats"]["code_normalization_reasons"] == {
        "identity_utf8_source_preserved": 3
    }

    packaged = []
    for name in ("train.jsonl", "validation.jsonl"):
        packaged.extend(
            json.loads(line)
            for line in (tmp_path / "out" / name).read_text(encoding="utf-8").splitlines()
        )
    assert len(packaged) == 3
    for item in packaged:
        assert item["modality"] == "code"
        assert item["raw_content_sha256"] == item["content_sha256"]
        assert item["normalization"]["source_sha256"] == item["content_sha256"]
        assert item["normalization"]["normalized_sha256"] == item["content_sha256"]
        assert item["normalization"]["transformations"] == []

    assignments = manifest["document_assignments"]
    assert len(assignments) == 3
    assert all(item["raw_content_sha256"] == item["normalized_content_sha256"] for item in assignments)
