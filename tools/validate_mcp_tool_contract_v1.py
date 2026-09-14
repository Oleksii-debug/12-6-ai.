from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.tool_protocol import ToolContractError, validate_contract_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MCP Tool Contract V1 manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/postbase/mcp_tool_contract_v1.json"),
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        identity = validate_contract_manifest(manifest)
    except (OSError, json.JSONDecodeError, ToolContractError) as exc:
        print(f"MCP Tool Contract V1: FAIL: {exc}")
        return 1
    print("MCP Tool Contract V1: PASS")
    print(f"manifest_identity={identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
