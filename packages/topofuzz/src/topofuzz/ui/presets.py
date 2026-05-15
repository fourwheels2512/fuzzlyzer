"""UI workflow + thoroughness presets.

A `Workflow` ties the user-facing label to a real fixture file in
examples/gallery/ and a curated set of starter scenarios. A `Thoroughness`
selects variants count, judge model, and the raw topofuzz cost cap (the
UI divides by 2.75 before showing real dollars to the user).

Both are baked in — no user customization. Per the minimalist-scope
feedback memory: would a solo founder ship this in 1-2 weekends?
Combinatorial config UIs are not a 1-weekend ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Repo root — five levels up from this file
# (packages/topofuzz/src/topofuzz/ui/presets.py → repo root):
#   parents[0] = ui/
#   parents[1] = topofuzz/        (the python package)
#   parents[2] = src/
#   parents[3] = topofuzz/        (the distribution dir under packages/)
#   parents[4] = packages/
#   parents[5] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
_GALLERY = _REPO_ROOT / "examples" / "gallery"


@dataclass(frozen=True)
class StarterScenario:
    """A pre-filled scenario card the user can edit or remove."""

    case: str
    why_borderline: str
    expected: str  # "approve" | "deny"


@dataclass(frozen=True)
class Workflow:
    """A domain workflow = one fixture + one rubric + starter scenarios."""

    key: str
    label: str
    short_desc: str
    domain_noun: str  # e.g. "loan application", "insurance claim"
    decision_verb: str  # e.g. "approve / deny"
    fixture_path: Path
    rubric_path: Path
    starters: tuple[StarterScenario, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Thoroughness:
    """Quick & cheap / Balanced demo / Premium proof."""

    key: str
    label: str
    short_desc: str
    variants: int
    judge_model: str
    # Raw topofuzz max-cost-usd. The UI divides by COST_DIVISOR before
    # showing real dollars (see cost.py).
    raw_cost_cap_usd: float
    # Realistic minute estimate (human-facing).
    typical_minutes: int


# --- Loan-approval starter scenarios -----------------------------------
_LOAN_STARTERS: tuple[StarterScenario, ...] = (
    StarterScenario(
        case=(
            "Applicant requests a $24,000 unsecured personal loan over 5 years. "
            "Annual income $78,000, salaried, current job held for 10 months; "
            "applicant worked continuously in the same field at a prior employer "
            "before that. Credit score 712. Existing monthly debt payments total "
            "$1,750 against gross monthly income of $6,500 (DTI 27%). Savings "
            "balance $9,000, about 5 months of debt service. No late payments in "
            "6 years of credit history."
        ),
        why_borderline=(
            "Job tenure at the current employer is under one year. Core ratios, "
            "score, and reserves are otherwise sound."
        ),
        expected="deny",
    ),
    StarterScenario(
        case=(
            "Applicant requests a $20,000 home improvement loan over 7 years. "
            "Annual income $120,000, salaried, same employer 12 years. Credit "
            "score 786. Existing monthly debt payments $1,670 against gross "
            "monthly income of $10,000 (DTI 17%). Savings and investment "
            "balances total $58,000. Owns home with $215,000 equity. No late "
            "payments in 8 years of credit history."
        ),
        why_borderline=(
            "Negligible. DTI 17%, deep reserves, high score, large equity "
            "cushion — a strong-approve stabilizer."
        ),
        expected="approve",
    ),
    StarterScenario(
        case=(
            "Applicant requests a $18,000 loan over 4 years to fund a small "
            "retail business expansion. Annual income $71,000 from the "
            "applicant's salaried role of 6 years; spouse not employed. Credit "
            "score 712. Existing monthly debt payments $1,775 against gross "
            "monthly income of $5,916 (DTI 30%; salary alone covers the new "
            "payment). Savings balance $9,500, about 5 months of debt service. "
            "The business is 16 months old. No late payments in 7 years."
        ),
        why_borderline=(
            "Stated purpose ties to a business with only 16 months of operating "
            "history; single-earner household."
        ),
        expected="deny",
    ),
)


# --- Insurance-claim starter scenarios ---------------------------------
_INSURANCE_STARTERS: tuple[StarterScenario, ...] = (
    StarterScenario(
        case=(
            "Member, age 47, requesting reimbursement of $4,200 for a "
            "lumbar-spine MRI. Member has been on the plan for 9 months. The "
            "MRI was ordered by the member's primary-care physician after three "
            "office visits over the past six months for lower-back pain. Member "
            "has met the annual deductible. The imaging facility is in-network. "
            "Pre-authorization paperwork was filed the day after the scan."
        ),
        why_borderline=(
            "Order came from primary-care rather than a specialist; pre-auth "
            "filed retroactively. Each element is a soft policy concern."
        ),
        expected="deny",
    ),
    StarterScenario(
        case=(
            "Member, age 62, requesting reimbursement of $1,850 for a routine "
            "annual physical and standard preventive blood-panel labs at an "
            "in-network primary-care office. Member has been on the plan for 6 "
            "years. All services are explicitly listed as covered preventive "
            "care under the policy with $0 copay. Pre-authorization not "
            "required. All documentation submitted on time."
        ),
        why_borderline=(
            "Negligible — preventive care explicitly covered, long-tenured "
            "member, in-network."
        ),
        expected="approve",
    ),
)


WORKFLOWS: dict[str, Workflow] = {
    "loan-approval": Workflow(
        key="loan-approval",
        label="Loan approval",
        short_desc="4 agents decide whether to approve a loan",
        domain_noun="loan application",
        decision_verb="approve / deny",
        fixture_path=_GALLERY / "langgraph-adversarial-relay" / "fixture.py",
        rubric_path=_GALLERY / "langgraph-adversarial-relay" / "rubric.txt",
        starters=_LOAN_STARTERS,
    ),
    "insurance-claim": Workflow(
        key="insurance-claim",
        label="Insurance claim triage",
        short_desc="4 agents decide whether to pay an insurance claim",
        domain_noun="insurance claim",
        decision_verb="approve / deny",
        fixture_path=_GALLERY / "insurance-claim-triage" / "fixture.py",
        rubric_path=_GALLERY / "insurance-claim-triage" / "rubric.txt",
        starters=_INSURANCE_STARTERS,
    ),
}


THOROUGHNESS: dict[str, Thoroughness] = {
    "quick": Thoroughness(
        key="quick",
        label="Quick & cheap",
        short_desc="fewer orderings, cheaper judge — for a first look",
        variants=2,
        judge_model="google/gemini-2.5-flash",
        raw_cost_cap_usd=2.20,
        typical_minutes=3,
    ),
    "balanced": Thoroughness(
        key="balanced",
        label="Balanced demo",
        short_desc="more orderings + smart judge — the demo configuration",
        variants=4,
        judge_model="google/gemini-2.5-pro",
        raw_cost_cap_usd=8.00,
        typical_minutes=7,
    ),
    "premium": Thoroughness(
        key="premium",
        label="Most thorough",
        short_desc="most orderings + smart judge — fullest evidence",
        variants=8,
        judge_model="google/gemini-2.5-pro",
        raw_cost_cap_usd=23.10,
        typical_minutes=12,
    ),
}


DEFAULT_WORKFLOW = "loan-approval"
DEFAULT_THOROUGHNESS = "balanced"


def get_workflow(key: str) -> Workflow:
    """Return the workflow by key, raising KeyError if unknown."""
    return WORKFLOWS[key]


def get_thoroughness(key: str) -> Thoroughness:
    """Return the thoroughness preset by key, raising KeyError if unknown."""
    return THOROUGHNESS[key]
