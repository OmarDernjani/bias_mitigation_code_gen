import sys

from .metrics import evaluate
from .runner import run_experiment


def _usage() -> None:
    print("usage:")
    print("  python -m bias_mitigation run")
    print("  python -m bias_mitigation eval <results.json> [out.json]")


def main() -> None:
    if len(sys.argv) < 2:
        _usage(); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "run":
        run_experiment()
    elif cmd == "eval":
        if len(sys.argv) < 3:
            _usage(); sys.exit(1)
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        _usage(); sys.exit(1)


if __name__ == "__main__":
    main()
