# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""`python -m fuzzeval` smoke entrypoint."""

from fuzzeval import __version__


def main() -> None:
    print(f"fuzzeval {__version__}")


if __name__ == "__main__":
    main()
