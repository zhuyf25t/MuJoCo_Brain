"""Re-evaluate one recorded capability input without moving a robot."""

import argparse
import json
from pathlib import Path

from .capabilities import Capabilities, load_profile
from .contracts import CapabilityInput
from .memory import RunLog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/langgraph-replay"))
    args = parser.parse_args(argv)
    records = (json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines())
    record = next((r for r in records if r["event"] == "capability_input" and r["call_id"] == args.call_id), None)
    if record is None:
        parser.error("No capability_input with that call-id")
    capabilities = Capabilities(load_profile(args.profile))
    log = RunLog(args.out)
    capabilities.begin_frame(log)
    try:
        result = capabilities.call(record["capability"], CapabilityInput.from_recorded(record["input"]))
        print(result.model_dump_json(indent=2))
        print(f"Recorded comparison: {log.path}")
    finally:
        capabilities.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
