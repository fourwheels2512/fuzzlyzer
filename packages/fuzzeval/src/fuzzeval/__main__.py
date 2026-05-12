"""`python -m fuzzeval` smoke entrypoint.

Real modules (judge, stats, cache, attestation, costest) are added in
Weekend 1 step 2 and beyond per BLUEPRINT_v5plus.md §14.
"""

from fuzzeval import __version__


def main() -> None:
    print(f"fuzzeval {__version__} — v5+ build in progress.")
    print("Planned modules: judge, stats, cache, attestation, costest.")
    print("See BLUEPRINT_v5plus.md §17a for the frozen public API contract.")


if __name__ == "__main__":
    main()
