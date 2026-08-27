from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.faiss_retrieval_qualification import build_evidence, probe_faiss


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the FAISS retrieval qualification V1 contract.")
    parser.add_argument("contract", type=Path)
    parser.add_argument("--probe-faiss", action="store_true", help="Run the optional local FAISS fixture probe.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    evidence = build_evidence(contract)
    if args.probe_faiss:
        evidence["local_faiss_probe"] = probe_faiss(contract)
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
