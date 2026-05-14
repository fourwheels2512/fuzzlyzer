"""fuzzeval — shared eval infrastructure for topofuzz (and siblings).

Public API per BLUEPRINT_v5plus.md §17a. All listed symbols are
implemented as of v0.1.0 (released alongside topofuzz v0.5.0rc1).
Importing the package is intentionally cheap so the verify-only path
stays under its dependency budget — no LangGraph and no LLM SDKs are
pulled at import time.

Side effect at import time: installs the secret-redaction filter on
the root logger (BLUEPRINT_v5plus.md §A "Structured logging" v0.5
commitment). Idempotent — calling install_default_filter() again is
a no-op. See fuzzeval.log_redaction for the patterns scrubbed.
"""

from fuzzeval.log_redaction import (
    SecretRedactionFilter,
    install_default_filter,
    uninstall_default_filter,
)

# Install before any other fuzzeval submodule emits its first log line.
install_default_filter()

from fuzzeval.attestation import (
    DSSE_PAE_PREFIX,
    DSSEDecodeError,
    DSSESignatureError,
    DSSEVerifyError,
    Ed25519Signer,
    Signer,
    VerifiedPayload,
    dsse_sign,
    dsse_verify,
    pae,
)
from fuzzeval.cache import JudgeCache, LLMCache
from fuzzeval.canonical import canonical_hash
from fuzzeval.costest import CostEstimate, CostLine, estimate
from fuzzeval.judge import (
    BinaryScore,
    ContinuousScore,
    Judge,
    JudgeResult,
    JudgeUnparseableError,
    SingleJudge,
    binary_from_continuous,
)
from fuzzeval.keys import (
    KeyAlreadyExistsError,
    KeyNotFoundError,
    default_dev_priv_path,
    default_dev_pub_path,
    generate_dev_keys,
    load_private_key,
    load_public_key,
)
from fuzzeval.stats import (
    BootstrapResult,
    percentile_bootstrap,
    percentile_cluster_bootstrap,
    wilson_ci,
)

__version__ = "0.1.0"

__all__ = [
    "BinaryScore",
    "BootstrapResult",
    "ContinuousScore",
    "CostEstimate",
    "CostLine",
    "DSSEDecodeError",
    "DSSESignatureError",
    "DSSEVerifyError",
    "DSSE_PAE_PREFIX",
    "Ed25519Signer",
    "Judge",
    "JudgeCache",
    "JudgeResult",
    "JudgeUnparseableError",
    "KeyAlreadyExistsError",
    "KeyNotFoundError",
    "LLMCache",
    "SecretRedactionFilter",
    "Signer",
    "SingleJudge",
    "VerifiedPayload",
    "__version__",
    "binary_from_continuous",
    "canonical_hash",
    "default_dev_priv_path",
    "default_dev_pub_path",
    "dsse_sign",
    "dsse_verify",
    "estimate",
    "generate_dev_keys",
    "install_default_filter",
    "load_private_key",
    "load_public_key",
    "pae",
    "percentile_bootstrap",
    "percentile_cluster_bootstrap",
    "wilson_ci",
]
