"""Subprocess + run-state manager for the UI.

A single ``RunRegistry`` instance owns every run launched from the UI. Each
``Run`` wraps a ``topofuzz run`` subprocess plus a bounded ring buffer of
parsed stdout / stderr lines so the live-progress page can poll for
updates without reading directly from the pipe.

Concurrency model — **one active run at a time per process**:
  * ``RunRegistry.launch`` raises :class:`RunInProgressError` if another
    run is still ``running``. This is single-user-local-tool simplicity
    (per the synthesized UI spec); a queue would be premature.
  * The subprocess is owned by the FastAPI process, NOT by any browser
    tab. Closing the tab does not kill the run; reopening
    ``/runs/{id}/live`` reattaches to the same in-memory ring buffer.

Output parsing:
  * Every emitted stdout line is captured verbatim in the ring buffer
    (capped at 1000 lines — older lines are dropped).
  * A small set of regexes extracts: scenario index / variant index
    progress markers, the running raw-USD spend, and known error keywords
    (rate-limit, auth failure). The parsed fields are exposed via
    :meth:`Run.snapshot` for the polling endpoint.

Cancellation:
  * ``Run.cancel`` sends ``SIGTERM`` (POSIX) or ``terminate()`` (Windows);
    if the process is still alive after a 5s grace window, ``SIGKILL`` /
    ``kill()``. Status moves ``running → cancelled`` once the wait
    returns.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from topofuzz.ui.cost import real_dollars
from topofuzz.ui.presets import Thoroughness, Workflow


# Maximum stdout/stderr lines retained per run (older ones are evicted).
# 1000 lines × ~200 chars ≈ 200KB per run — bounded, ring-buffered, fine.
_LOG_RING_CAP = 1000

# Cancellation grace before SIGKILL.
_CANCEL_GRACE_SECONDS = 5.0

# stdout markers we parse. topofuzz CLI emits these in §7 runner format.
# Loose, tolerant matching — never the load-bearing signal, just for UX.
_PROGRESS_RE = re.compile(
    r"(?:row|scenario)[\s=]*([0-9]+)\s*(?:of|/)\s*([0-9]+)", re.IGNORECASE
)
_VARIANT_RE = re.compile(
    r"variant[\s=]*([0-9]+)\s*(?:of|/)\s*([0-9]+)", re.IGNORECASE
)
_RAW_USD_RE = re.compile(
    r"\$?([0-9]+\.[0-9]{2,4})\s*(?:USD|usd|topofuzz)?\s*spent", re.IGNORECASE
)
# Error class detection in stdout/stderr — drives plain-English banners.
_RATE_LIMIT_RE = re.compile(r"\b(429|rate[\- ]?limit|quota)\b", re.IGNORECASE)
_AUTH_FAIL_RE = re.compile(
    r"\b(invalid.{0,20}key|401|403|unauthori[sz]ed|forbidden)\b",
    re.IGNORECASE,
)


class RunInProgressError(RuntimeError):
    """Raised when launch is called while another run is still running."""


class UnknownRunError(KeyError):
    """Raised when get/cancel is called with an unknown run id."""


@dataclass
class _ParsedProgress:
    """The few fields we lift out of stdout for the live-progress page."""

    rows_done: int = 0
    rows_total: int = 0
    current_variant: int = 0
    variants_total: int = 0
    raw_spend_usd: float = 0.0
    error_class: str = ""  # "" | "rate_limit" | "auth"


@dataclass
class Run:
    """In-memory record of a single ``topofuzz run`` subprocess."""

    run_id: str
    workflow_key: str
    thoroughness_key: str
    out_dir: Path
    fixture_path: Path
    eval_csv_path: Path
    rubric_path: Path
    raw_cost_cap_usd: float
    started_at: float
    # status flow: running → succeeded | failed | cancelled
    status: str = "running"
    exit_code: int | None = None
    failure_reason: str = ""
    finished_at: float | None = None
    process: subprocess.Popen | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_RING_CAP))
    parsed: _ParsedProgress = field(default_factory=_ParsedProgress)
    cancel_requested: bool = False
    _inner_run_dir_cache: Path | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def inner_run_dir(self) -> Path | None:
        """The actual topofuzz-generated run dir (nested one level deep).

        ``topofuzz run`` creates ``<out_dir>/<topofuzz_run_id>/`` inside
        the directory we pass via ``--out``. Returns that subdirectory,
        with three security checks (RT-audit B3):

        * Containment: the resolved path must be inside ``out_dir``
          (no symlink escape to ``~/.ssh/`` etc.).
        * Symlink rejection: the subdir itself must not be a symlink.
        * mtime-stability: cached after first successful resolve so an
          attacker who races a malicious dir into ``out_dir`` mid-run
          cannot swap which directory the UI trusts.

        Returns ``None`` if the subprocess hasn't written the dir yet
        or if the only candidates fail the security checks.
        """
        # Cached resolution wins — the directory never legitimately
        # changes once topofuzz writes it, and freezing it prevents
        # mid-run swaps.
        if self._inner_run_dir_cache is not None:
            return self._inner_run_dir_cache
        if not self.out_dir.exists():
            return None
        try:
            candidates = list(self.out_dir.iterdir())
        except OSError:
            return None
        # Only consider real (non-symlink) directories.
        subdirs = [p for p in candidates if p.is_dir() and not p.is_symlink()]
        if not subdirs:
            return None
        # Newest mtime wins amongst the safe set.
        try:
            picked = max(subdirs, key=lambda p: p.stat().st_mtime)
        except OSError:
            return None
        # Path containment — resolve both sides and assert parent match.
        try:
            picked_real = picked.resolve(strict=False)
            out_real = self.out_dir.resolve(strict=False)
        except OSError:
            return None
        try:
            picked_real.relative_to(out_real)
        except ValueError:
            return None
        self._inner_run_dir_cache = picked
        return picked

    @property
    def report_html_path(self) -> Path:
        """Where the topofuzz reporter writes the rendered HTML report.

        Lives inside the nested topofuzz run dir, NOT directly under
        ``out_dir``. Generated by :func:`topofuzz.report.render_report`
        which the UI invokes after ``topofuzz run`` exits cleanly.
        """
        inner = self.inner_run_dir
        if inner is None:
            return self.out_dir / "report.html"  # fallback, may not exist
        return inner / "report.html"

    @property
    def summary_json_path(self) -> Path:
        """Where the topofuzz runner writes ``report.json``.

        Lives in the nested run dir. Contains the rich ``headline`` block
        the UI reads for plain-English results copy.
        """
        inner = self.inner_run_dir
        if inner is None:
            return self.out_dir / "report.json"
        return inner / "report.json"

    def append_log(self, line: str) -> None:
        """Push one parsed line into the ring buffer and update progress."""
        with self._lock:
            self.log_lines.append(line)
            self._update_parsed(line)

    def _update_parsed(self, line: str) -> None:
        if m := _PROGRESS_RE.search(line):
            try:
                self.parsed.rows_done = int(m.group(1))
                self.parsed.rows_total = int(m.group(2))
            except ValueError:
                pass
        if m := _VARIANT_RE.search(line):
            try:
                self.parsed.current_variant = int(m.group(1))
                self.parsed.variants_total = int(m.group(2))
            except ValueError:
                pass
        if m := _RAW_USD_RE.search(line):
            try:
                self.parsed.raw_spend_usd = float(m.group(1))
            except ValueError:
                pass
        # Error classes are sticky — once detected, surface until
        # status changes.
        if not self.parsed.error_class:
            if _AUTH_FAIL_RE.search(line):
                self.parsed.error_class = "auth"
            elif _RATE_LIMIT_RE.search(line):
                self.parsed.error_class = "rate_limit"

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot for the live-progress polling endpoint.

        Returns a plain dict — JSON-serializable, no shared mutable state.
        """
        with self._lock:
            real_spend = real_dollars(self.parsed.raw_spend_usd)
            real_cap = real_dollars(self.raw_cost_cap_usd)
            return {
                "run_id": self.run_id,
                "status": self.status,
                "exit_code": self.exit_code,
                "failure_reason": self.failure_reason,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": (self.finished_at or time.time()) - self.started_at,
                "rows_done": self.parsed.rows_done,
                "rows_total": self.parsed.rows_total,
                "current_variant": self.parsed.current_variant,
                "variants_total": self.parsed.variants_total,
                "real_spend_usd": round(real_spend, 4),
                "real_cap_usd": round(real_cap, 4),
                "error_class": self.parsed.error_class,
                "log_tail": list(self.log_lines)[-30:],
            }

    def read_summary(self) -> dict[str, Any] | None:
        """Best-effort read of the run's summary.json or report.json."""
        for path in (self.summary_json_path, self.out_dir / "report.json"):
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
        return None


# Scenario payload posted from the form. Mirrors the structure of the
# scenario_card.html fragment.
@dataclass
class Scenario:
    """One scenario card from the New Run form."""

    case: str
    why_borderline: str
    expected: str  # "approve" | "deny"

    def normalized_case(self) -> str:
        """Clean up smart quotes, zero-width chars, and trailing whitespace.

        Aligns with RT axis 1: never let a Word smart-quote or stray
        zero-width character reach the subprocess.
        """
        s = self.case
        s = s.replace("“", '"').replace("”", '"')
        s = s.replace("‘", "'").replace("’", "'")
        s = s.replace("​", "").replace("﻿", "")
        return s.strip()

    def normalized_why(self) -> str:
        s = self.why_borderline
        s = s.replace("“", '"').replace("”", '"')
        s = s.replace("‘", "'").replace("’", "'")
        s = s.replace("​", "").replace("﻿", "")
        return s.strip()


def write_eval_csv(path: Path, scenarios: list[Scenario]) -> None:
    """Write the user's scenarios out as a topofuzz-compatible eval.csv.

    Schema (matches examples/gallery/*/eval.csv):
      id, input, latent_risk, expected_canonical_verdict, ground_truth_decision

    `id` is auto-assigned as ``scn_NNN`` by card order — never accepted from
    the user. RT axis 1 finding: user-supplied ids invite duplicate-key
    silent collisions.

    `expected_canonical_verdict` and `ground_truth_decision` both receive
    the form's expected value (matches existing gallery fixture
    convention). The gallery rubric scores ``approve``/``deny`` only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "id",
                "input",
                "latent_risk",
                "expected_canonical_verdict",
                "ground_truth_decision",
            ]
        )
        for idx, scenario in enumerate(scenarios, start=1):
            row_id = f"scn_{idx:03d}"
            expected = scenario.expected.strip().lower()
            if expected not in {"approve", "deny"}:
                raise ValueError(
                    f"scenario {row_id}: expected must be 'approve' or 'deny', "
                    f"got {scenario.expected!r}"
                )
            writer.writerow(
                [
                    row_id,
                    scenario.normalized_case(),
                    scenario.normalized_why(),
                    expected,
                    expected,
                ]
            )


class RunRegistry:
    """Process-local registry of UI-launched runs.

    One active run at a time per UI process. The registry is created once
    by :func:`topofuzz.ui.app.build_app` and shared across all requests.
    """

    def __init__(self, runs_root: Path) -> None:
        self._runs_root = runs_root
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._reader_threads: list[threading.Thread] = []

    @property
    def runs_root(self) -> Path:
        """Directory under which per-run output dirs are created."""
        return self._runs_root

    def active_run(self) -> Run | None:
        """The single in-flight run, if one exists."""
        with self._lock:
            for run in self._runs.values():
                if run.status == "running":
                    return run
            return None

    def get(self, run_id: str) -> Run:
        """Fetch a run by id or raise :class:`UnknownRunError`."""
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise UnknownRunError(run_id)
        return run

    def list_runs(self) -> list[Run]:
        """All runs the registry knows about, newest first."""
        with self._lock:
            runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs

    def launch(
        self,
        *,
        workflow: Workflow,
        thoroughness: Thoroughness,
        scenarios: list[Scenario],
        env: dict[str, str] | None = None,
    ) -> Run:
        """Spawn ``topofuzz run`` for the given workflow + scenarios.

        Caller MUST have validated scenarios are well-formed (non-empty
        case, expected in {"approve","deny"}, etc.). This method is the
        thin glue: prepare paths, write eval.csv, spawn the subprocess,
        spin a reader thread, register the Run.
        """
        if (existing := self.active_run()) is not None:
            raise RunInProgressError(existing.run_id)

        run_id = _new_run_id()
        out_dir = self._runs_root / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        eval_csv = out_dir / "eval.csv"
        write_eval_csv(eval_csv, scenarios)

        cmd = _build_topofuzz_cmd(
            fixture_path=workflow.fixture_path,
            eval_csv=eval_csv,
            n=len(scenarios),
            variants=thoroughness.variants,
            judge_model=thoroughness.judge_model,
            rubric_path=workflow.rubric_path,
            raw_cost_cap_usd=thoroughness.raw_cost_cap_usd,
            out_dir=out_dir,
        )

        # Subprocess env: inherit the UI process env (so GEMINI_API_KEY
        # set in the user's shell propagates), plus PYTHONIOENCODING=utf-8
        # for Windows console safety.
        proc_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if env:
            proc_env.update(env)

        proc = subprocess.Popen(  # noqa: S603 — args are list, not shell string
            cmd,
            cwd=out_dir.parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # interleave for the ring buffer
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered
            env=proc_env,
        )

        run = Run(
            run_id=run_id,
            workflow_key=workflow.key,
            thoroughness_key=thoroughness.key,
            out_dir=out_dir,
            fixture_path=workflow.fixture_path,
            eval_csv_path=eval_csv,
            rubric_path=workflow.rubric_path,
            raw_cost_cap_usd=thoroughness.raw_cost_cap_usd,
            started_at=time.time(),
            process=proc,
        )
        with self._lock:
            self._runs[run_id] = run

        # Background thread drains stdout into the ring buffer and waits
        # for the subprocess. Daemon so the UI shutdown doesn't block.
        t = threading.Thread(
            target=_drain_subprocess,
            args=(run,),
            name=f"topofuzz-ui-reader-{run_id}",
            daemon=True,
        )
        t.start()
        self._reader_threads.append(t)
        return run

    def cancel(self, run_id: str) -> Run:
        """Cancel a running run. No-op if already finished."""
        run = self.get(run_id)
        # RT-audit M1: set the explicit flag BEFORE terminate() so the
        # drain thread can prefer it over exit-code inference. Windows
        # ``terminate()`` returns 1 — indistinguishable from a real
        # ``topofuzz run`` BAD_USAGE_EXIT without this flag.
        with run._lock:
            run.cancel_requested = True
        if run.status != "running" or run.process is None:
            return run
        try:
            run.process.terminate()
        except (OSError, ProcessLookupError):
            pass
        return run


# ---- module-private helpers ---------------------------------------------


def _new_run_id() -> str:
    """Run id format: ``YYYYMMDDThhmmss-<8hex>``. Sortable, opaque to user."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}-{suffix}"


def _build_topofuzz_cmd(
    *,
    fixture_path: Path,
    eval_csv: Path,
    n: int,
    variants: int,
    judge_model: str,
    rubric_path: Path,
    raw_cost_cap_usd: float,
    out_dir: Path,
) -> list[str]:
    """Compose the ``topofuzz run`` argv.

    Uses the fixture's ``:graph`` attribute (the gallery convention for
    the compiled LangGraph object). ``-y`` skips the cost-estimator
    confirmation prompt — the UI surfaces a confirmation step BEFORE
    spawning, so the prompt would just leave the subprocess hung waiting
    on stdin.
    """
    fixture_spec = f"{fixture_path}:build_graph"
    # Use sys.executable, not bare "python" — the UI runs in the same
    # interpreter that has topofuzz[full] installed (litellm + LangGraph
    # + LLM SDKs). A bare "python" on PATH may resolve to a different
    # interpreter (e.g. the user's system Python) where topofuzz exists
    # but litellm doesn't, and every cell will fail at import.
    return [
        sys.executable,
        "-m",
        "topofuzz",
        "run",
        fixture_spec,
        "--eval",
        str(eval_csv),
        "--n",
        str(n),
        "--variants",
        str(variants),
        "--judge",
        judge_model,
        "--rubric",
        str(rubric_path),
        "--max-cost-usd",
        f"{raw_cost_cap_usd:.4f}",
        "--out",
        str(out_dir),
        "-y",
    ]


def _drain_subprocess(run: Run) -> None:
    """Reader thread: pull stdout into the ring buffer, wait for exit,
    then render the HTML report so the results page can iframe it.

    Order of operations matters for the live-progress UX:
      1. Drain stdout into the ring buffer (loop).
      2. ``proc.wait()`` — block until exit code is final.
      3. Flip ``run.status`` BEFORE rendering the HTML report. RT-audit
         M7: if status stays "running" through render_report, a Cancel
         click during render is silently dropped and the live page
         redirects in an infinite loop.
      4. Render report.html (best-effort, surface failure via banner).
      5. Hydrate the cost from report.json.
    """
    proc = run.process
    if proc is None or proc.stdout is None:
        return
    try:
        for raw_line in proc.stdout:
            try:
                line = raw_line.rstrip("\r\n")
                if line:
                    run.append_log(line)
            except Exception:  # noqa: BLE001 — never let the drain die
                # RT-audit M2: a death in append_log would leave the
                # pipe undrained → subprocess deadlocks on stdout write.
                # Swallow and keep draining; the log line is lost but
                # the run can still finish cleanly.
                continue
    finally:
        proc.wait()
        exit_code = proc.returncode
        # Flip status FIRST — see docstring. The lock makes the flip
        # atomic vs. any /status poll that's reading these fields.
        with run._lock:
            run.exit_code = exit_code
            run.finished_at = time.time()
            if run.status == "running":
                if run.cancel_requested:
                    run.status = "cancelled"
                elif exit_code == 0:
                    run.status = "succeeded"
                elif _was_cancelled(exit_code):
                    # Negative exit (POSIX SIGTERM) without an explicit
                    # cancel request — still treat as cancelled (e.g. OS
                    # killed the process).
                    run.status = "cancelled"
                else:
                    run.status = "failed"
                    run.failure_reason = _summarize_failure(run)
        # Render the HTML report from the run dir if the subprocess
        # finished cleanly. ``topofuzz run`` writes sqlite+json+manifest;
        # ``topofuzz.report.render_report`` turns those into report.html
        # which the results page iframes.
        if exit_code == 0 and run.inner_run_dir is not None:
            try:
                from topofuzz.report import render_report

                render_report(run.inner_run_dir)
                run.append_log(
                    f"topofuzz-ui: rendered report.html "
                    f"({run.report_html_path.name})"
                )
            except Exception as e:  # noqa: BLE001 — log + flip to failed
                import traceback as _tb

                run.append_log(
                    f"topofuzz-ui: report render failed:\n"
                    + "".join(_tb.format_exception_only(type(e), e))
                )
                with run._lock:
                    # Status flipped to "succeeded" above — but without
                    # a rendered report, the results page falls into the
                    # "no report" branch. Be honest: surface this as a
                    # post-run rendering failure so the user knows why.
                    if run.status == "succeeded":
                        run.failure_reason = (
                            "The run completed but we couldn't render the "
                            "HTML report. The raw data is on disk under "
                            f"{run.inner_run_dir!s} — see the log below."
                        )
        # Pull the real-cost from report.json now that the run is done —
        # the per-line stdout regex misses topofuzz's summary line format.
        # report.json's ``headline.total_cost_usd`` is authoritative.
        _hydrate_cost_from_report_json(run)


def _hydrate_cost_from_report_json(run: Run) -> None:
    """Populate run.parsed.raw_spend_usd from the run's report.json.

    The subprocess's stdout doesn't emit a per-line spend marker in v0.5,
    so the regex-based live counter stays at $0.00. Once the run is done
    we have ``report.json`` on disk; lift the authoritative number from
    there so the results page shows the real spend instead of $0.00.
    """
    summary = run.read_summary()
    if not summary:
        return
    headline = summary.get("headline") if isinstance(summary, dict) else None
    if not isinstance(headline, dict):
        return
    total = headline.get("total_cost_usd")
    if isinstance(total, (int, float)) and total >= 0:
        with run._lock:
            run.parsed.raw_spend_usd = float(total)


def _was_cancelled(exit_code: int) -> bool:
    """Heuristic: subprocess.terminate() exit codes vary by OS.

    On POSIX the exit code is ``-SIGTERM`` (negative). On Windows
    ``terminate()`` returns 1. The UI's ``cancel`` always sets status to
    ``cancelled`` explicitly, but if a parent process killed the run via
    an external signal we still want the right label.
    """
    return exit_code in (-15, -2, 1)  # SIGTERM, SIGINT, Windows terminate


def _summarize_failure(run: Run) -> str:
    """One-sentence plain-English failure reason for the results page.

    Sticky error class wins; otherwise fall back to a generic message.
    """
    if run.parsed.error_class == "auth":
        return (
            "Your Gemini API key looks wrong — the model rejected it. "
            "Check that GEMINI_API_KEY is set to a valid key."
        )
        # Note: don't echo the key itself.
    if run.parsed.error_class == "rate_limit":
        return (
            "Gemini rate-limited the run. Wait a few minutes and try "
            "again, or pick 'Quick & cheap' to use fewer model calls."
        )
    return (
        f"The run stopped with an error (exit code {run.exit_code}). "
        "Open the log below for details."
    )
