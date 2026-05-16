# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for fuzzeval.log_redaction.SecretRedactionFilter.

Covers each pattern type, args interpolation (positional + mapping),
non-string passthrough, idempotent install, and the never-drop
invariant (filter.filter() must always return True).
"""

from __future__ import annotations

import io
import logging

import pytest

from fuzzeval.log_redaction import (
    SecretRedactionFilter,
    install_default_filter,
    uninstall_default_filter,
)


@pytest.fixture
def filter_instance() -> SecretRedactionFilter:
    return SecretRedactionFilter()


@pytest.fixture
def captured_logger() -> tuple[logging.Logger, io.StringIO]:
    """Yield a fresh logger + an in-memory handler for output assertion."""
    logger = logging.getLogger("test_log_redaction_isolated")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Tear down between tests
    for h in list(logger.handlers):
        logger.removeHandler(h)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(SecretRedactionFilter())
    logger.addHandler(handler)
    return logger, buf


class TestPatternMatching:
    def test_openai_sk_key_redacted(self, filter_instance):
        text = "API call with token sk-proj-abc123_xyz456-DEF789ghi failed"
        scrubbed = filter_instance._scrub(text)
        assert "sk-proj" not in scrubbed
        assert "[REDACTED-OPENAI-KEY]" in scrubbed

    def test_xai_key_redacted(self, filter_instance):
        text = "header: xai-1234567890abcdefghij_KLM"
        scrubbed = filter_instance._scrub(text)
        assert "xai-1234" not in scrubbed
        assert "[REDACTED-XAI-KEY]" in scrubbed

    def test_aws_access_key_redacted(self, filter_instance):
        text = "creds: AKIAIOSFODNN7EXAMPLE for s3 access"
        scrubbed = filter_instance._scrub(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "[REDACTED-AWS-ACCESS-KEY]" in scrubbed

    def test_google_aistudio_key_redacted(self, filter_instance):
        # AIza prefix + exactly 35 chars after = 39 total. Use a realistic shape.
        text = "x-goog-api-key: AIzaSyEXAMPLETESTKEYNOTREAL000000000001"
        scrubbed = filter_instance._scrub(text)
        assert "AIzaSyEXAMPLETESTKEYNOTREAL000000000001" not in scrubbed
        assert "[REDACTED-GOOGLE-API-KEY]" in scrubbed

    def test_bearer_token_redacted(self, filter_instance):
        text = "Authorization: Bearer abc123def456ghi789jkl012mno345pqr"
        scrubbed = filter_instance._scrub(text)
        assert "abc123def456" not in scrubbed
        assert "Bearer [REDACTED-TOKEN]" in scrubbed

    def test_multiple_secrets_in_one_string_all_redacted(self, filter_instance):
        text = (
            "Failed retry: sk-proj-AAAAAAAAAAAAAAAAAAAA "
            "and AKIAIOSFODNN7EXAMPLE both rejected"
        )
        scrubbed = filter_instance._scrub(text)
        assert "[REDACTED-OPENAI-KEY]" in scrubbed
        assert "[REDACTED-AWS-ACCESS-KEY]" in scrubbed
        assert "sk-proj" not in scrubbed
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed


class TestNoFalsePositives:
    def test_short_sk_prefix_below_threshold_not_matched(self, filter_instance):
        # "sk-" + 19 chars is below the 20-char minimum body
        text = "ski-resort lodge sk-tooshort1234567890"
        scrubbed = filter_instance._scrub(text)
        assert scrubbed == text

    def test_aiza_prefix_with_wrong_length_not_matched(self, filter_instance):
        # AIza + only 10 chars (not the required 35)
        text = "AIzaShortKey"
        scrubbed = filter_instance._scrub(text)
        assert scrubbed == text

    def test_safe_message_passes_through_unchanged(self, filter_instance):
        text = "Cell completed: variant=abc123 eval=lending_001 score=0.85"
        scrubbed = filter_instance._scrub(text)
        assert scrubbed == text

    def test_canonical_hash_not_matched(self, filter_instance):
        # JCS canonical hash is 64 hex chars but doesn't carry the
        # provider prefixes, so it survives.
        text = (
            "topology_canonical_hash="
            "d2faf66a23cdb0f06785433c4c565f99457b63ae2a95cba873fee6bb0be8417c"
        )
        scrubbed = filter_instance._scrub(text)
        assert scrubbed == text


class TestArgsInterpolation:
    def test_positional_args_redacted(self, captured_logger):
        logger, buf = captured_logger
        logger.warning("API failed with %r", "sk-proj-AAAAAAAAAAAAAAAAAAAAA")
        out = buf.getvalue()
        assert "sk-proj" not in out
        assert "[REDACTED-OPENAI-KEY]" in out

    def test_mapping_args_redacted(self, captured_logger):
        logger, buf = captured_logger
        logger.warning("got %(token)s", {"token": "Bearer xyz789abc456def012"})
        out = buf.getvalue()
        assert "xyz789" not in out
        assert "[REDACTED-TOKEN]" in out

    def test_non_string_args_passthrough(self, captured_logger):
        logger, buf = captured_logger
        logger.warning("count=%d cost=%.2f", 42, 1.234)
        out = buf.getvalue()
        assert "count=42" in out
        assert "cost=1.23" in out

    def test_mixed_args_only_string_redacted(self, captured_logger):
        logger, buf = captured_logger
        logger.warning(
            "cell=%d token=%r", 7, "sk-proj-BBBBBBBBBBBBBBBBBBBBB"
        )
        out = buf.getvalue()
        assert "cell=7" in out
        assert "sk-proj" not in out
        assert "[REDACTED-OPENAI-KEY]" in out


class TestNeverDropInvariant:
    def test_filter_always_returns_true_on_safe_message(self, filter_instance):
        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="safe", args=(), exc_info=None,
        )
        assert filter_instance.filter(rec) is True

    def test_filter_always_returns_true_on_secret_message(self, filter_instance):
        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="leaked sk-proj-CCCCCCCCCCCCCCCCCCCC", args=(), exc_info=None,
        )
        assert filter_instance.filter(rec) is True
        # Side-effect: msg got rewritten in place
        assert "sk-proj" not in rec.msg

    def test_non_string_msg_passthrough_no_crash(self, filter_instance):
        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg={"event": "structured"}, args=(), exc_info=None,
        )
        assert filter_instance.filter(rec) is True
        assert rec.msg == {"event": "structured"}


class TestInstallation:
    def setup_method(self):
        # Ensure clean state across test_log_redaction.py tests in a
        # single pytest run; install_default_filter is run at fuzzeval
        # import time, so the filter is already on the root logger.
        uninstall_default_filter()

    def teardown_method(self):
        # Restore the package-default install state for the rest of
        # the test suite.
        install_default_filter()

    def test_install_returns_true_on_first_call(self):
        assert install_default_filter() is True

    def test_install_returns_false_on_second_call(self):
        install_default_filter()
        assert install_default_filter() is False

    def test_uninstall_returns_true_when_present(self):
        install_default_filter()
        assert uninstall_default_filter() is True

    def test_uninstall_returns_false_when_absent(self):
        # setup_method() already uninstalled
        assert uninstall_default_filter() is False

    def test_root_logger_has_filter_after_install(self):
        install_default_filter()
        root = logging.getLogger()
        assert any(
            isinstance(f, SecretRedactionFilter) for f in root.filters
        )

    def test_root_logger_no_filter_after_uninstall(self):
        install_default_filter()
        uninstall_default_filter()
        root = logging.getLogger()
        assert not any(
            isinstance(f, SecretRedactionFilter) for f in root.filters
        )


def test_fuzzeval_import_installs_filter():
    """Importing fuzzeval should auto-install the redaction filter."""
    # The package was already imported by other tests in this run; the
    # marker should be set on the root logger.
    root = logging.getLogger()
    assert getattr(root, "_fuzzeval_log_redaction_installed", False) is True
