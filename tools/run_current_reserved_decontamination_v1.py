"""Secure entrypoint for current reserved-evaluation decontamination.

The execution core remains the audited PR1821 implementation.  This entrypoint adds a
mechanically derived executable-closure gate: before any payload-bearing read it
compares the loaded implementation's behavior-bearing globals and attribute bindings
with the same exact checkout loaded in a fresh ``python -I`` interpreter.
"""
from __future__ import annotations

import ast
import dis
import hashlib
import json
import marshal
import math
import os
import re
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

from tools import _run_current_reserved_decontamination_v1_core as _core

matching_impl = _core.matching_impl
current_impl = _core.current_impl
authority_impl = _core.authority_impl

RECEIPT_SCHEMA = _core.RECEIPT_SCHEMA
REPORT_NAME = _core.REPORT_NAME
EVIDENCE_NAME = _core.EVIDENCE_NAME
RECEIPT_NAME = _core.RECEIPT_NAME

_CORE_PUBLISH_BUNDLE = _core._publish_bundle
_CORE_RENAME_DIRECTORY_NO_REPLACE = _core._rename_directory_no_replace
_rename_directory_no_replace = _CORE_RENAME_DIRECTORY_NO_REPLACE

_IMPLEMENTATION_MODULES = (
    (
        "tools/run_current_reserved_decontamination_v1.py",
        "tools.run_current_reserved_decontamination_v1",
    ),
    (
        "tools/_run_current_reserved_decontamination_v1_core.py",
        "tools._run_current_reserved_decontamination_v1_core",
    ),
    (
        "src/twelve_six/data/current_reserved_decontamination_v1.py",
        "twelve_six.data.current_reserved_decontamination_v1",
    ),
    (
        "src/twelve_six/data/decontamination_authority_v2.py",
        "twelve_six.data.decontamination_authority_v2",
    ),
    (
        "src/twelve_six/data/_data232_decontamination_matching.py",
        "twelve_six.data._data232_decontamination_matching",
    ),
)
_GLOBAL_LOAD_OPS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
_ATTRIBUTE_LOAD_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})


def __getattr__(name: str) -> object:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


def _normalized_code(code: types.CodeType) -> types.CodeType:
    constants = tuple(
        _normalized_code(item) if isinstance(item, types.CodeType) else item
        for item in code.co_consts
    )
    return code.replace(co_filename="<bound-source>", co_consts=constants)


def _code_sha256(code: types.CodeType) -> str:
    return hashlib.sha256(marshal.dumps(_normalized_code(code))).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
    return f"{module}.{qualname}"


def _describe_value(value: object, *, depth: int = 0) -> object:
    if depth > 12:
        raise RuntimeError("behavior-closure value nesting is unexpectedly deep")
    exact_type = type(value)
    if value is None or exact_type in {bool, int, str}:
        return {
            "kind": "scalar",
            "type": _type_name(value),
            "value": value,
        }
    if exact_type is float:
        if not math.isfinite(value):
            return {
                "kind": "float",
                "type": _type_name(value),
                "value": repr(value),
            }
        return {
            "kind": "float",
            "type": _type_name(value),
            "value": value,
        }
    if exact_type is bytes:
        return {
            "kind": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    pattern_type = type(re.compile(""))
    if exact_type is pattern_type:
        return {
            "kind": "regex",
            "pattern": value.pattern,
            "flags": int(value.flags),
        }
    if isinstance(value, types.ModuleType):
        spec = getattr(value, "__spec__", None)
        return {
            "kind": "module",
            "name": value.__name__,
            "file": getattr(value, "__file__", None),
            "origin": getattr(spec, "origin", None),
        }
    if isinstance(value, types.FunctionType):
        return {
            "kind": "python_function",
            "name": _callable_name(value),
            "code_sha256": _code_sha256(value.__code__),
            "defaults": _describe_value(value.__defaults__, depth=depth + 1),
            "kwdefaults": _describe_value(value.__kwdefaults__, depth=depth + 1),
        }
    builtin_callable_types = (
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
    )
    if isinstance(value, builtin_callable_types):
        return {
            "kind": "builtin_callable",
            "type": _type_name(value),
            "name": _callable_name(value),
        }
    if isinstance(value, type):
        return {
            "kind": "type",
            "name": _callable_name(value),
            "bases": [
                f"{base.__module__}.{base.__qualname__}" for base in value.__bases__
            ],
        }
    if exact_type in {tuple, list}:
        return {
            "kind": exact_type.__name__,
            "items": [
                _describe_value(item, depth=depth + 1) for item in value
            ],
        }
    if exact_type in {set, frozenset}:
        items = [_describe_value(item, depth=depth + 1) for item in value]
        return {
            "kind": exact_type.__name__,
            "items": sorted(items, key=_canonical_json),
        }
    if exact_type is dict:
        items = [
            (
                _describe_value(key, depth=depth + 1),
                _describe_value(child, depth=depth + 1),
            )
            for key, child in value.items()
        ]
        items.sort(key=lambda item: _canonical_json(item[0]))
        return {
            "kind": "dict",
            "items": items,
        }
    if isinstance(value, Mapping):
        items = [
            (
                _describe_value(key, depth=depth + 1),
                _describe_value(child, depth=depth + 1),
            )
            for key, child in value.items()
        ]
        items.sort(key=lambda item: _canonical_json(item[0]))
        return {
            "kind": "mapping",
            "type": _type_name(value),
            "items": items,
        }
    raise RuntimeError(
        "unsupported behavior-bearing binding type: "
        f"{_type_name(value)}"
    )


def _describe_attribute_value(value: object) -> object:
    """Describe behavior-relevant attributes without serializing ambient registries."""
    exact_type = type(value)
    pattern_type = type(re.compile(""))
    safe = (
        value is None
        or exact_type in {bool, int, float, str, bytes}
        or exact_type is pattern_type
        or isinstance(value, types.ModuleType)
        or isinstance(value, types.FunctionType)
        or isinstance(
            value,
            (
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
            ),
        )
        or isinstance(value, type)
    )
    if safe:
        return _describe_value(value)
    return {
        "kind": "attribute_object",
        "type": _type_name(value),
    }


def _walk_code_objects(code: types.CodeType):
    yield code
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            yield from _walk_code_objects(item)


def _referenced_bindings(function: types.FunctionType) -> dict[str, object]:
    result: dict[str, object] = {}
    builtins = function.__builtins__
    if isinstance(builtins, types.ModuleType):
        builtin_map = vars(builtins)
    else:
        builtin_map = builtins

    for code in _walk_code_objects(function.__code__):
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in _GLOBAL_LOAD_OPS:
                continue
            name = instruction.argval
            if not isinstance(name, str):
                continue
            if name in function.__globals__:
                scope = "global"
                value = function.__globals__[name]
            elif name in builtin_map:
                scope = "builtin"
                value = builtin_map[name]
            else:
                result[f"missing:{name}"] = {"kind": "missing"}
                continue
            result[f"{scope}:{name}"] = _describe_value(value)

            chain: list[str] = []
            current = value
            cursor = index + 1
            while (
                cursor < len(instructions)
                and instructions[cursor].opname in _ATTRIBUTE_LOAD_OPS
            ):
                attribute = instructions[cursor].argval
                if not isinstance(attribute, str):
                    break
                chain.append(attribute)
                try:
                    current = getattr(current, attribute)
                except Exception as exc:
                    result[f"{scope}:{name}." + ".".join(chain)] = {
                        "kind": "attribute_error",
                        "type": _type_name(exc),
                    }
                    break
                result[f"{scope}:{name}." + ".".join(chain)] = (
                    _describe_attribute_value(current)
                )
                cursor += 1
    return dict(sorted(result.items()))


def _top_level_function_names(source: str, path: Path) -> tuple[str, ...]:
    tree = ast.parse(source, filename=str(path))
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return tuple(sorted(names))


def _module_snapshot(module: object, source_path: Path) -> dict[str, object]:
    source = source_path.read_text(encoding="utf-8")
    compiled = compile(
        source,
        str(source_path),
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    expected_codes = {
        item.co_name: item
        for item in compiled.co_consts
        if isinstance(item, types.CodeType)
    }
    functions: dict[str, object] = {}
    for name in _top_level_function_names(source, source_path):
        actual = getattr(module, name, None)
        expected = expected_codes.get(name)
        if not isinstance(actual, types.FunctionType) or expected is None:
            raise RuntimeError(
                f"loaded implementation callable set drift: "
                f"{getattr(module, '__name__', '?')}.{name}"
            )
        if _code_sha256(actual.__code__) != _code_sha256(expected):
            raise RuntimeError(
                f"loaded implementation callable code drift: "
                f"{getattr(module, '__name__', '?')}.{name}"
            )
        functions[name] = {
            "code_sha256": _code_sha256(actual.__code__),
            "defaults": _describe_value(actual.__defaults__),
            "kwdefaults": _describe_value(actual.__kwdefaults__),
            "bindings": _referenced_bindings(actual),
        }
    return {
        "module_name": getattr(module, "__name__", None),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "functions": functions,
    }


def _loaded_module(relative: str, module_name: str) -> object:
    if relative == "tools/run_current_reserved_decontamination_v1.py":
        return sys.modules[__name__]
    if relative == "tools/_run_current_reserved_decontamination_v1_core.py":
        return _core
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(f"implementation module is not loaded: {module_name}")
    return module


def _require_loaded_sources_from_checkout(repo_root: Path) -> None:
    root = repo_root.resolve()
    for relative, module_name in _IMPLEMENTATION_MODULES:
        module = _loaded_module(relative, module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(
                f"loaded implementation source has no origin: {relative}"
            )
        checkout_path = root / relative
        expected_path = checkout_path.resolve()
        actual_path = Path(module_file).resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                "loaded implementation source is not from the exact checkout: "
                f"{relative} expected={expected_path} actual={actual_path}"
            )
        if checkout_path.is_symlink() or not expected_path.is_file():
            raise RuntimeError(
                "loaded implementation source is not a regular checkout file: "
                f"{relative}"
            )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                "loaded implementation source is not tracked at expected head: "
                f"{relative}"
            )
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if blob.returncode != 0 or not blob.stdout.strip():
            raise RuntimeError(
                "unable to bind implementation source blob at expected head: "
                f"{relative}"
            )


def _behavior_snapshot(repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    modules: dict[str, object] = {}
    for relative, module_name in _IMPLEMENTATION_MODULES:
        module = _loaded_module(relative, module_name)
        modules[relative] = _module_snapshot(module, root / relative)
    return {
        "schema": "12-6.current-reserved-executable-closure.v1",
        "python": {
            "implementation": sys.implementation.name,
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
            "optimize": sys.flags.optimize,
        },
        "modules": modules,
    }


def _probe_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "LANG",
        "LC_ALL",
    )
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _emit_behavior_probe(repo_root_text: str, expected_git_sha: str) -> None:
    repo_root = Path(repo_root_text)
    actual = require_exact_checkout(repo_root, expected_git_sha)
    if actual != expected_git_sha:
        raise RuntimeError("isolated probe Git head drift")
    _require_loaded_sources_from_checkout(repo_root)
    print(_canonical_json(_behavior_snapshot(repo_root)))


def _isolated_behavior_snapshot(
    repo_root: Path,
    expected_git_sha: str,
) -> dict[str, object]:
    root = repo_root.resolve()
    bootstrap = (
        "import importlib,sys;"
        "root=sys.argv[1];"
        "sys.path[0:0]=[root,root+'/src'];"
        "m=importlib.import_module("
        "'tools.run_current_reserved_decontamination_v1');"
        "m._emit_behavior_probe(root,sys.argv[2])"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(root),
            expected_git_sha,
        ],
        cwd=root,
        env=_probe_environment(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise RuntimeError(
            "isolated executable-closure probe failed"
            + (f": {detail}" if detail else "")
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("isolated executable-closure probe output drift")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "isolated executable-closure probe emitted invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("isolated executable-closure probe root drift")
    return value


def require_exact_checkout(repo_root: Path, expected_git_sha: str) -> str:
    return _core.require_exact_checkout(repo_root, expected_git_sha)


def require_exact_implementation(repo_root: Path, expected_git_sha: str) -> str:
    actual = require_exact_checkout(repo_root, expected_git_sha)
    _require_loaded_sources_from_checkout(repo_root)
    local = _behavior_snapshot(repo_root)
    isolated = _isolated_behavior_snapshot(repo_root, expected_git_sha)
    if local != isolated:
        raise RuntimeError(
            "isolated executable behavior closure drift before payload access"
        )
    return actual


def _publish_bundle(output_dir: Path, files: Mapping[str, bytes]) -> None:
    original = _core._rename_directory_no_replace
    try:
        _core._rename_directory_no_replace = _rename_directory_no_replace
        _CORE_PUBLISH_BUNDLE(output_dir, files)
    finally:
        _core._rename_directory_no_replace = original


def _authority_args(argv: list[str] | None) -> tuple[Path, str]:
    raw = list(sys.argv[1:] if argv is None else argv)

    def read_flag(name: str, *, default: str | None = None) -> str:
        values: list[str] = []
        index = 0
        prefix = name + "="
        while index < len(raw):
            item = raw[index]
            if item == name:
                if index + 1 >= len(raw):
                    raise ValueError(f"{name} requires a value")
                values.append(raw[index + 1])
                index += 2
                continue
            if item.startswith(prefix):
                values.append(item[len(prefix) :])
            index += 1
        if len(values) > 1:
            raise ValueError(f"{name} must be supplied at most once")
        if values:
            return values[0]
        if default is not None:
            return default
        raise ValueError(f"{name} is required")

    return (
        Path(read_flag("--repo-root", default=".")),
        read_flag("--expected-decontamination-implementation-git-sha"),
    )


def main(argv: list[str] | None = None) -> int:
    repo_root, expected_git_sha = _authority_args(argv)

    # This exact-code + isolated behavior-closure gate intentionally precedes the
    # core parser and every payload-bearing input read.
    actual = require_exact_implementation(repo_root, expected_git_sha)

    original_gate = _core.require_exact_implementation
    try:
        _core.require_exact_implementation = lambda *_args, **_kwargs: actual
        return _core.main(argv)
    finally:
        _core.require_exact_implementation = original_gate


if __name__ == "__main__":
    raise SystemExit(main())
