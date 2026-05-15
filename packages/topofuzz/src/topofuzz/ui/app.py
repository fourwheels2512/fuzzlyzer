"""FastAPI app — the 5th-grader-runnable UI.

Routes:
  GET  /                       → landing (one button: "Test your AI pipeline")
  GET  /setup                  → New Run form (preset + scenarios + cost)
  POST /setup                  → kicks off a run, redirects to /runs/{id}/live
  GET  /setup/cost-preview     → HTMX fragment: re-render cost row
  GET  /setup/example          → HTMX fragment: re-seed scenario cards on preset change
  POST /setup/scenario         → HTMX fragment: append a blank scenario card
  POST /setup/scenario/remove  → HTMX fragment: drop the last card (returns blank)
  GET  /runs                   → flat history list
  GET  /runs/{id}/live         → live progress page (auto-polls)
  GET  /runs/{id}/status       → HTMX poll endpoint (JSON-like fragment)
  POST /runs/{id}/cancel       → kill the subprocess, redirect to results
  GET  /runs/{id}              → done page: plain-English headline + iframe
  GET  /runs/{id}/report       → serves report.html from the run dir
  GET  /healthz                → "ok" (used by the launcher to know we're up)

Single-run-at-a-time. A second POST /setup while one is running returns
422 with a friendly "wait for the current run to finish or cancel it"
banner. This is the simplest concurrency model that doesn't surprise a
5th grader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from topofuzz import __version__
from topofuzz.ui.cost import format_dollars, real_dollars
from topofuzz.ui.presets import (
    DEFAULT_THOROUGHNESS,
    DEFAULT_WORKFLOW,
    THOROUGHNESS,
    WORKFLOWS,
    StarterScenario,
    Thoroughness,
    Workflow,
    get_thoroughness,
    get_workflow,
)
from topofuzz.ui.runs import (
    RunInProgressError,
    RunRegistry,
    Scenario,
    UnknownRunError,
)


_THIS_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _THIS_DIR / "templates"
_STATIC_DIR = _THIS_DIR / "static"


def _default_runs_root() -> Path:
    """Where UI-launched runs are written. ~/.topofuzz/ui-runs/ by default."""
    home = Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    return home / ".topofuzz" / "ui-runs"


def build_app(*, runs_root: Path | None = None) -> FastAPI:
    """Construct the FastAPI app. Factory style so tests can inject a
    temp ``runs_root`` and avoid touching ``~/.topofuzz/``.
    """
    runs_root = runs_root or _default_runs_root()
    runs_root.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(runs_root=runs_root)

    app = FastAPI(
        title="topofuzz UI",
        version=__version__,
        docs_url=None,  # local-only tool, no need to expose OpenAPI docs
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=_TEMPLATES_DIR)
    templates.env.globals["format_dollars"] = format_dollars
    templates.env.globals["topofuzz_version"] = __version__

    # ---------- helpers --------------------------------------------------

    def _gemini_key_present() -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def _scenarios_from_form(form: dict[str, Any]) -> list[Scenario]:
        """Parse posted scenario_case[]/scenario_why[]/scenario_expected[]
        into Scenario dataclasses. The form template emits parallel lists
        keyed by suffix ``[]`` so the order is preserved.

        Lists are tolerant: a card with empty `case` is dropped (RT axis 1
        — never let blank-or-junk reach the subprocess).
        """
        cases = form.getlist("scenario_case") if hasattr(form, "getlist") else []
        whys = form.getlist("scenario_why") if hasattr(form, "getlist") else []
        expecteds = (
            form.getlist("scenario_expected") if hasattr(form, "getlist") else []
        )
        # Pad shorter lists so zip doesn't silently truncate.
        n = max(len(cases), len(whys), len(expecteds))
        scenarios: list[Scenario] = []
        for i in range(n):
            case = (cases[i] if i < len(cases) else "").strip()
            why = (whys[i] if i < len(whys) else "").strip()
            expected = (expecteds[i] if i < len(expecteds) else "").strip().lower()
            if not case:
                continue
            if expected not in {"approve", "deny"}:
                expected = "deny"  # safe default; user can edit
            scenarios.append(
                Scenario(case=case, why_borderline=why, expected=expected)
            )
        return scenarios

    def _cost_preview(
        *, workflow_key: str, thoroughness_key: str, n_scenarios: int
    ) -> dict[str, Any]:
        """Compute the headline numbers shown on the form's cost row.

        The CLI's real cost depends on prompt length × variants × judge
        cost — none of which the UI can predict precisely. We use the
        thoroughness preset's `raw_cost_cap_usd` (already a budget cap,
        not a measured spend) divided by 2.75, and SCALE LINEARLY by
        scenario count vs the preset's natural 10. This gives a number
        within a factor of ~2 of reality — good enough for the form's
        "around $X" surface, never load-bearing.
        """
        thoroughness = get_thoroughness(thoroughness_key)
        # Defensive — if `WORKFLOWS` ever changes, fall back to default.
        if workflow_key not in WORKFLOWS:
            workflow_key = DEFAULT_WORKFLOW
        scale = max(1, n_scenarios) / 10.0
        real_cap = real_dollars(thoroughness.raw_cost_cap_usd) * scale
        return {
            "real_cap_usd": real_cap,
            "real_cap_fmt": format_dollars(real_cap),
            "minutes_typical": max(1, int(thoroughness.typical_minutes * scale)),
            "n_scenarios": n_scenarios,
        }

    def _starters_as_scenarios(workflow: Workflow) -> list[StarterScenario]:
        return list(workflow.starters)

    # ---------- routes ---------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    async def landing(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "landing.html",
            {"key_present": _gemini_key_present()},
        )

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_get(
        request: Request,
        workflow: str = DEFAULT_WORKFLOW,
        thoroughness: str = DEFAULT_THOROUGHNESS,
    ) -> HTMLResponse:
        if not _gemini_key_present():
            return templates.TemplateResponse(
                request, "no_key.html", {}, status_code=200
            )
        workflow_obj = WORKFLOWS.get(workflow) or WORKFLOWS[DEFAULT_WORKFLOW]
        thoroughness_obj = THOROUGHNESS.get(thoroughness) or THOROUGHNESS[DEFAULT_THOROUGHNESS]
        starters = _starters_as_scenarios(workflow_obj)
        # Pad to 5 cards by default so the page feels prepared rather than empty.
        while len(starters) < 5:
            starters.append(
                StarterScenario(case="", why_borderline="", expected="deny")
            )
        cost = _cost_preview(
            workflow_key=workflow_obj.key,
            thoroughness_key=thoroughness_obj.key,
            n_scenarios=len(starters),
        )
        active = registry.active_run()
        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "workflows": list(WORKFLOWS.values()),
                "thoroughnesses": list(THOROUGHNESS.values()),
                "selected_workflow": workflow_obj,
                "selected_thoroughness": thoroughness_obj,
                "starters": starters,
                "cost": cost,
                "active_run_id": active.run_id if active else None,
            },
        )

    @app.post("/setup", response_model=None)
    async def setup_post(request: Request):  # noqa: ANN202 — union response
        form = await request.form()
        workflow_key = (form.get("workflow") or DEFAULT_WORKFLOW).strip()
        thoroughness_key = (form.get("thoroughness") or DEFAULT_THOROUGHNESS).strip()
        try:
            workflow_obj = get_workflow(workflow_key)
        except KeyError:
            workflow_obj = WORKFLOWS[DEFAULT_WORKFLOW]
        try:
            thoroughness_obj = get_thoroughness(thoroughness_key)
        except KeyError:
            thoroughness_obj = THOROUGHNESS[DEFAULT_THOROUGHNESS]

        scenarios = _scenarios_from_form(form)
        # Minimum-scenario floor: with fewer than 3, the noise floor swamps
        # any signal. RT axis 2 — refuse with an inline error rather than
        # spawning a meaningless run.
        if len(scenarios) < 3:
            cost = _cost_preview(
                workflow_key=workflow_obj.key,
                thoroughness_key=thoroughness_obj.key,
                n_scenarios=max(len(scenarios), 1),
            )
            return templates.TemplateResponse(
                request,
                "setup.html",
                {
                    "workflows": list(WORKFLOWS.values()),
                    "thoroughnesses": list(THOROUGHNESS.values()),
                    "selected_workflow": workflow_obj,
                    "selected_thoroughness": thoroughness_obj,
                    "starters": _hydrate_starters_from_form(scenarios),
                    "cost": cost,
                    "form_error": (
                        "Write at least three scenarios. With fewer than that, "
                        "the result isn't trustworthy enough to demo."
                    ),
                    "active_run_id": None,
                },
                status_code=400,
            )

        if not _gemini_key_present():
            return templates.TemplateResponse(
                request, "no_key.html", {}, status_code=400
            )

        try:
            run = registry.launch(
                workflow=workflow_obj,
                thoroughness=thoroughness_obj,
                scenarios=scenarios,
            )
        except RunInProgressError as exc:
            return templates.TemplateResponse(
                request,
                "setup.html",
                {
                    "workflows": list(WORKFLOWS.values()),
                    "thoroughnesses": list(THOROUGHNESS.values()),
                    "selected_workflow": workflow_obj,
                    "selected_thoroughness": thoroughness_obj,
                    "starters": _hydrate_starters_from_form(scenarios),
                    "cost": _cost_preview(
                        workflow_key=workflow_obj.key,
                        thoroughness_key=thoroughness_obj.key,
                        n_scenarios=len(scenarios),
                    ),
                    "form_error": (
                        "Another run is already going. Wait for it to finish "
                        "or cancel it before starting a new one."
                    ),
                    "active_run_id": str(exc),
                },
                status_code=409,
            )

        return RedirectResponse(url=f"/runs/{run.run_id}/live", status_code=303)

    @app.get("/setup/cost-preview", response_class=HTMLResponse)
    async def cost_preview(
        request: Request,
        workflow: str = DEFAULT_WORKFLOW,
        thoroughness: str = DEFAULT_THOROUGHNESS,
        n_scenarios: int = 5,
    ) -> HTMLResponse:
        cost = _cost_preview(
            workflow_key=workflow,
            thoroughness_key=thoroughness,
            n_scenarios=n_scenarios,
        )
        return templates.TemplateResponse(
            request, "fragments/cost_row.html", {"cost": cost}
        )

    @app.get("/setup/example", response_class=HTMLResponse)
    async def setup_example(
        request: Request, workflow: str = DEFAULT_WORKFLOW
    ) -> HTMLResponse:
        """Re-seed the scenario cards when the user switches workflow.

        Returns a list of fresh scenario_card fragments for the chosen
        workflow's starters, padded to 5 cards. The form template's
        scenario-card container target swaps with this fragment.
        """
        workflow_obj = WORKFLOWS.get(workflow) or WORKFLOWS[DEFAULT_WORKFLOW]
        starters = _starters_as_scenarios(workflow_obj)
        while len(starters) < 5:
            starters.append(
                StarterScenario(case="", why_borderline="", expected="deny")
            )
        return templates.TemplateResponse(
            request, "fragments/scenario_cards.html", {"starters": starters}
        )

    @app.post("/setup/scenario", response_class=HTMLResponse)
    async def add_scenario_card(request: Request) -> HTMLResponse:
        """Return one fresh empty scenario card to append to the form."""
        return templates.TemplateResponse(
            request,
            "fragments/scenario_card.html",
            {
                "starter": StarterScenario(
                    case="", why_borderline="", expected="deny"
                ),
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_history(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "history.html", {"runs": registry.list_runs()}
        )

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    async def live(request: Request, run_id: str) -> HTMLResponse:
        try:
            run = registry.get(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        # If the run is already done, redirect to the results page so
        # refresh-on-done lands the user in the right place.
        if run.status != "running":
            return RedirectResponse(url=f"/runs/{run_id}", status_code=303)
        snap = run.snapshot()
        workflow_obj = WORKFLOWS.get(run.workflow_key) or WORKFLOWS[DEFAULT_WORKFLOW]
        thoroughness_obj = THOROUGHNESS.get(run.thoroughness_key) or THOROUGHNESS[DEFAULT_THOROUGHNESS]
        return templates.TemplateResponse(
            request,
            "live.html",
            {
                "run": run,
                "snap": snap,
                "workflow": workflow_obj,
                "thoroughness": thoroughness_obj,
            },
        )

    @app.get("/runs/{run_id}/status", response_class=HTMLResponse)
    async def run_status(request: Request, run_id: str) -> HTMLResponse:
        try:
            run = registry.get(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        snap = run.snapshot()
        workflow_obj = WORKFLOWS.get(run.workflow_key) or WORKFLOWS[DEFAULT_WORKFLOW]
        return templates.TemplateResponse(
            request,
            "fragments/live_status.html",
            {
                "run": run,
                "snap": snap,
                "workflow": workflow_obj,
            },
        )

    @app.post("/runs/{run_id}/cancel")
    async def run_cancel(run_id: str) -> RedirectResponse:
        try:
            registry.cancel(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_results(request: Request, run_id: str) -> HTMLResponse:
        try:
            run = registry.get(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        if run.status == "running":
            return RedirectResponse(url=f"/runs/{run_id}/live", status_code=303)
        snap = run.snapshot()
        summary = run.read_summary()
        workflow_obj = WORKFLOWS.get(run.workflow_key) or WORKFLOWS[DEFAULT_WORKFLOW]
        headline = _build_headline(run=run, summary=summary, workflow=workflow_obj)
        report_exists = run.report_html_path.exists()
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "run": run,
                "snap": snap,
                "summary": summary,
                "workflow": workflow_obj,
                "headline": headline,
                "report_exists": report_exists,
            },
        )

    @app.get("/runs/{run_id}/report")
    async def run_report(run_id: str) -> FileResponse:
        try:
            run = registry.get(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        if not run.report_html_path.exists():
            raise HTTPException(status_code=404, detail="report not yet written")
        # RT-audit B2 reinforcement: even with iframe sandbox, lock the
        # report content down with a strict CSP. Inline styles are needed
        # for the report's SVG charts; everything else is denied. data:
        # is allowed for inlined SVG images.
        return FileResponse(
            run.report_html_path,
            media_type="text/html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'none'; "
                    "style-src 'unsafe-inline'; "
                    "img-src data: 'self'; "
                    "font-src data: 'self'; "
                    "frame-ancestors 'self'"
                ),
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    return app


def _hydrate_starters_from_form(
    scenarios: list[Scenario],
) -> list[StarterScenario]:
    """Preserve what the user typed when re-rendering the form on error."""
    out = [
        StarterScenario(
            case=s.case, why_borderline=s.why_borderline, expected=s.expected
        )
        for s in scenarios
    ]
    while len(out) < 5:
        out.append(StarterScenario(case="", why_borderline="", expected="deny"))
    return out


def _build_headline(
    *,
    run: "Any",
    summary: dict[str, Any] | None,
    workflow: Workflow,
) -> dict[str, Any]:
    """Plain-English headline for the results page.

    Reads from topofuzz's ``report.json`` schema:
      * ``headline.verdict_flip_rate.point`` — fraction of (variant, eval)
        cells whose verdict differs from the original ordering. Multiplied
        by ``n_evals`` to get a "X of N" plain-English headline.
      * ``headline.confirmed_verdict_flip_rate`` — populated only after a
        ``topofuzz confirm`` pass; otherwise None.
      * ``headline.n_evals`` — number of eval rows in this run.
      * ``headline.judge_failure_rate`` — fraction of judge calls that
        failed to extract APPROVE/DENY. Surfaced as a "the AI couldn't
        decide on N answers" caveat when significant.

    Falls back to a generic "your run completed" line when the summary
    can't be parsed.
    """
    headline_d = summary.get("headline") if isinstance(summary, dict) else None
    if not isinstance(headline_d, dict):
        headline_d = {}

    n_total = headline_d.get("n_evals")
    if not isinstance(n_total, int) or n_total <= 0:
        # Fallback — count rows in our written eval csv.
        try:
            with run.eval_csv_path.open(encoding="utf-8") as f:
                n_total = max(0, sum(1 for _ in f) - 1)
        except OSError:
            n_total = 0

    # Confirmed verdict-flip rate (post-noise-floor confirm pass).
    confirmed_rate = headline_d.get("confirmed_verdict_flip_rate")
    confirmed_count: int | None
    if isinstance(confirmed_rate, (int, float)) and n_total:
        confirmed_count = round(confirmed_rate * n_total)
    else:
        confirmed_count = None

    # Raw verdict-flip rate from the run (always present on a clean run).
    raw_rate_d = headline_d.get("verdict_flip_rate")
    raw_count: int | None = None
    if isinstance(raw_rate_d, dict):
        point = raw_rate_d.get("point")
        if isinstance(point, (int, float)) and n_total:
            raw_count = round(point * n_total)

    # Judge-failure flag — degraded runs need a caveat.
    judge_failure_rate = headline_d.get("judge_failure_rate") or 0
    judge_failed_count = 0
    if isinstance(judge_failure_rate, (int, float)) and n_total:
        judge_failed_count = round(
            judge_failure_rate * n_total * (headline_d.get("n_variants") or 4)
        )

    if run.status in ("succeeded", "failed"):
        if confirmed_count is not None:
            return {
                "title": (
                    f"Out of {n_total} {_plural(workflow.domain_noun, n_total)}, "
                    f"your AI pipeline changed its mind on {confirmed_count} of them "
                    f"when we shuffled the order of its steps."
                ),
                "subtitle": (
                    "This is the noise-checked count — apparent flips that "
                    "could be the AI second-guessing itself have been removed."
                ),
                "flip_count": confirmed_count,
                "n_total": n_total,
            }
        if raw_count is not None:
            caveat = ""
            if judge_failed_count > 0:
                caveat = (
                    f" (Note: the AI couldn't read a clear yes-or-no from "
                    f"{judge_failed_count} of its own answers — see the full "
                    f"report.)"
                )
            return {
                "title": (
                    f"Out of {n_total} {_plural(workflow.domain_noun, n_total)}, "
                    f"your AI pipeline changed its mind on about {raw_count} of them "
                    f"when we shuffled the order of its steps."
                ),
                "subtitle": (
                    "We haven't run the optional noise check — some flips could "
                    "be the AI second-guessing itself rather than real "
                    "order-sensitivity." + caveat
                ),
                "flip_count": raw_count,
                "n_total": n_total,
            }
    # n_evals=0 / malformed report.json — be honest with the user.
    return {
        "title": "Your run finished, but we couldn't read its summary file.",
        "subtitle": (
            "This is unusual — the run's `report.json` was empty or "
            "malformed. Open the run log below; if it looks broken, this "
            "is a topofuzz bug worth reporting."
        ),
        "flip_count": None,
        "n_total": n_total,
    }


def _plural(noun: str, n: int) -> str:
    if n == 1:
        return noun
    if noun.endswith("y") and not noun.endswith("ay"):
        return noun[:-1] + "ies"
    if noun.endswith("s"):
        return noun
    return noun + "s"
