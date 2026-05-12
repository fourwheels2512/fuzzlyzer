"""`python -m fuzzeval` smoke entrypoint."""

from fuzzeval import __version__


def main() -> None:
    print(f"fuzzeval {__version__}")


if __name__ == "__main__":
    main()
