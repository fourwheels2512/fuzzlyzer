"""fuzzeval — shared eval infrastructure for topofuzz (and siblings).

Public API. Note: :func:`percentile_cluster_bootstrap` is a §17a contract
pin that lands in step 8; calling it before then raises
``NotImplementedError``. Importing it (and the rest of the package) at
module load is intentionally cheap so callers can wire the surface today
and rely on the implementation as it lands.
"""

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

__version__ = "0.1.0.dev0"

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
    "load_private_key",
    "load_public_key",
    "pae",
    "percentile_bootstrap",
    "percentile_cluster_bootstrap",
    "wilson_ci",
]
