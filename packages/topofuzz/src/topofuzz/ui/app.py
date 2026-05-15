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

import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from topofuzz import __version__
from topofuzz.ui.cost import format_dollars, real_dollars
from topofuzz.ui.presets import (
    ALLOWED_CUSTOM_MODELS,
    CUSTOM_WORKFLOW_KEY_PREFIX,
    DEFAULT_THOROUGHNESS,
    DEFAULT_WORKFLOW,
    THOROUGHNESS,
    WORKFLOWS,
    CustomWorkflow,
    StarterScenario,
    Thoroughness,
    Workflow,
    get_thoroughness,
    get_workflow,
)
from topofuzz.ui.custom_storage import (
    delete_custom_workflow,
    list_custom_workflows,
    load_custom_workflow,
    save_custom_workflow,
)
from topofuzz.ui.custom_validation import (
    CustomFormError,
    validate_custom_workflow_form,
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

# Maximum POST body the custom-workflow editor accepts. 64 KB is far above
# the legitimate form size (label + 10 agents + 2 KB rubric ≈ 3 KB) and
# below any value that could exhaust memory by request flood.
_CUSTOM_FORM_MAX_BYTES = 64 * 1024

# Strict CSP applied to both ``/runs/{id}/inspect`` outputs (the rendered
# inspect HTML and the "no flips" fallback). Identical to the report
# route's posture: inline styles only, nothing else.
_INSPECT_HEADERS: dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'none'; "
        "style-src 'unsafe-inline'; "
        "frame-ancestors 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Friendly empty-state page for ``GET /runs/{id}/inspect`` when the run
# had no verdict flips to inspect. A clean run is not an error condition;
# returning HTML 200 lets the iframe show a useful message instead of
# the browser's default 404 chrome.
_NO_FLIPS_HTML = (
    "<!DOCTYPE html>"
    "<html lang=\"en\"><head>"
    "<meta charset=\"utf-8\">"
    "<title>topofuzz inspect — no flips</title>"
    "<style>"
    "html,body{margin:0;background:#faf7f2;color:#2a2622;"
    "font-family:Inter,-apple-system,system-ui,sans-serif;}"
    ".wrap{max-width:560px;margin:80px auto;padding:32px;"
    "background:#fff;border:1px solid #e7d8c2;border-radius:12px;}"
    "h1{font-family:'Source Serif Pro',Georgia,serif;font-weight:500;"
    "font-size:24px;margin:0 0 10px;color:#1f1c19;}"
    "p{margin:0 0 8px;line-height:1.6;color:#6b5f54;}"
    "code{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;"
    "background:#f0eae0;padding:1px 6px;border-radius:3px;font-size:13px;}"
    "</style></head><body><div class=\"wrap\">"
    "<h1>Nothing to inspect on this run.</h1>"
    "<p>The inspector compares two orderings whose verdicts disagreed. "
    "On this run, no scenario produced a verdict flip across orderings — "
    "your pipeline was stable.</p>"
    "<p>To see this view in action, try a more borderline scenario, "
    "increase thoroughness, or open <code>topofuzz inspect</code> on a "
    "run that flips.</p>"
    "</div></body></html>"
)


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

    def _all_workflows_for_setup() -> list[Workflow]:
        """Built-ins (in their declared order) plus saved custom workflows.

        Custom workflows come last, sorted by ``created_at`` descending —
        i.e. the freshest custom workflow appears just after the built-ins.
        """
        out: list[Workflow] = list(WORKFLOWS.values())
        for cw in list_custom_workflows():
            try:
                out.append(_materialize_custom(cw))
            except (OSError, KeyError, ValueError) as exc:
                # Narrow except: never let one bad workflow break the setup
                # page, but log the failure so the user has something to
                # diagnose. Don't catch Exception — that would hide real
                # bugs in template rendering or schema construction.
                _log.warning(
                    "could not materialize custom workflow %s: %s",
                    cw.workflow_id,
                    exc,
                )
                continue
        return out

    def _materialize_custom(cw: CustomWorkflow) -> Workflow:
        """Resolve a CustomWorkflow into a runnable :class:`Workflow`.

        Goes through :func:`get_workflow` so the same on-disk path is
        exercised that ``POST /setup`` will hit when the user clicks Run.
        """
        return get_workflow(cw.key)

    def _resolve_workflow_or_default(key: str) -> Workflow:
        """Look up ``key`` (built-in OR ``custom:...``); fall back to default.

        Used in error-path renders where we just need *something* sensible
        to show the user — never raises.
        """
        try:
            return get_workflow(key)
        except KeyError:
            return WORKFLOWS[DEFAULT_WORKFLOW]

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
        all_workflows = _all_workflows_for_setup()
        # Resolve selected workflow against built-ins AND custom; fall back
        # to the default if the requested key has been deleted.
        workflow_obj: Workflow
        try:
            workflow_obj = get_workflow(workflow)
        except KeyError:
            workflow_obj = WORKFLOWS[DEFAULT_WORKFLOW]
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
                "workflows": all_workflows,
                "thoroughnesses": list(THOROUGHNESS.values()),
                "selected_workflow": workflow_obj,
                "selected_thoroughness": thoroughness_obj,
                "starters": starters,
                "cost": cost,
                "active_run_id": active.run_id if active else None,
                "custom_prefix": CUSTOM_WORKFLOW_KEY_PREFIX,
            },
        )

    @app.post("/setup", response_model=None)
    async def setup_post(request: Request):  # noqa: ANN202 — union response
        _csrf_guard(request)
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
                    "workflows": _all_workflows_for_setup(),
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
                    "workflows": _all_workflows_for_setup(),
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
        workflow_obj = _resolve_workflow_or_default(workflow)
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
        workflow_obj = _resolve_workflow_or_default(run.workflow_key)
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
        workflow_obj = _resolve_workflow_or_default(run.workflow_key)
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
    async def run_cancel(request: Request, run_id: str) -> RedirectResponse:
        _csrf_guard(request)
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
        workflow_obj = _resolve_workflow_or_default(run.workflow_key)
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

    @app.get("/runs/{run_id}/inspect")
    async def run_inspect(run_id: str) -> Response:
        """Render + serve the side-by-side flip inspector for one eval.

        Picks the first flipping eval deterministically (lex-order),
        writes ``inspect_<eid>.html`` into the run dir on demand, and
        serves it with the same strict CSP as ``/report``. When there
        are no flips to inspect we return a friendly 200 HTML "nothing
        to inspect" page rather than a 404 — a clean run is not an
        error condition (audit RT-pass-1 MINOR-4).
        """
        try:
            run = registry.get(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        inner = run.inner_run_dir
        if inner is None:
            raise HTTPException(
                status_code=404, detail="run output directory not yet written"
            )
        # Lazy import — inspect.py pulls jinja2/markupsafe (from the
        # ``[full]`` extra). Keeps the UI module importable in a verify-
        # only install.
        from topofuzz.inspect import InspectError, write_inspect_html

        try:
            target = write_inspect_html(inner)
        except InspectError:
            # Not a failure mode — the run had no verdict flips to show.
            return HTMLResponse(_NO_FLIPS_HTML, headers=_INSPECT_HEADERS)
        return FileResponse(
            target,
            media_type="text/html",
            headers=_INSPECT_HEADERS,
        )

    # ---- Custom workflow ("BYOP") routes -----------------------------

    def _render_custom_editor(
        request: Request,
        *,
        cw: CustomWorkflow | None,
        form_errors: dict[str, str] | None = None,
        form_values: dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        """Render the custom-workflow editor in create or edit mode.

        ``cw`` is the existing workflow being edited (None on create).
        ``form_values`` lets the route preserve a partially-filled form
        when validation fails — without it, the user retypes everything.
        """
        # Build the editor's "current state": prefer form_values (latest
        # user input that failed validation), then fall back to the saved
        # custom workflow, then to a sensible blank create-mode default.
        if form_values is not None:
            ctx_label = form_values.get("label", "")
            ctx_domain = form_values.get("domain_noun", "")
            ctx_verb = form_values.get("decision_verb", "")
            ctx_rubric = form_values.get("rubric_paragraph", "")
            ctx_agents = form_values.get(
                "agents", [("", ALLOWED_CUSTOM_MODELS[0][0])]
            )
            ctx_verdict = form_values.get("verdict_agent_id", "")
        elif cw is not None:
            ctx_label = cw.label
            ctx_domain = cw.domain_noun
            ctx_verb = cw.decision_verb
            ctx_rubric = cw.rubric_paragraph
            ctx_agents = [(a.agent_id, a.model) for a in cw.agents]
            ctx_verdict = cw.verdict_agent_id
        else:
            # Blank create-mode form: 4 starter empty rows.
            ctx_label = ""
            ctx_domain = "loan application"
            ctx_verb = "approve / deny"
            ctx_rubric = ""
            ctx_agents = [("", ALLOWED_CUSTOM_MODELS[0][0])] * 4
            ctx_verdict = ""
        return templates.TemplateResponse(
            request,
            "custom_workflow_editor.html",
            {
                "cw": cw,
                "label": ctx_label,
                "domain_noun": ctx_domain,
                "decision_verb": ctx_verb,
                "rubric_paragraph": ctx_rubric,
                "agents": ctx_agents,
                "verdict_agent_id": ctx_verdict,
                "models": ALLOWED_CUSTOM_MODELS,
                "form_errors": form_errors or {},
                "form_action": (
                    f"/custom-workflow/{cw.workflow_id}"
                    if cw is not None
                    else "/custom-workflow"
                ),
                "min_agents": 1,
                "max_agents": 10,
            },
            status_code=status_code,
        )

    def _form_values_from_post(form: Any) -> dict[str, Any]:
        """Extract form fields into the dict shape _render_custom_editor wants.

        Used to round-trip a failed POST back through the editor without
        losing the user's input.
        """
        agent_ids = (
            form.getlist("agent_id") if hasattr(form, "getlist") else []
        )
        agent_models = (
            form.getlist("agent_model") if hasattr(form, "getlist") else []
        )
        n = max(len(agent_ids), len(agent_models))
        agents = []
        for i in range(n):
            aid = (agent_ids[i] if i < len(agent_ids) else "").strip()
            mdl = (agent_models[i] if i < len(agent_models) else "").strip()
            agents.append((aid, mdl))
        return {
            "label": (form.get("label") or "").strip(),
            "domain_noun": (form.get("domain_noun") or "").strip(),
            "decision_verb": (form.get("decision_verb") or "").strip(),
            "rubric_paragraph": (form.get("rubric_paragraph") or "").strip(),
            "agents": agents or [("", ALLOWED_CUSTOM_MODELS[0][0])],
            "verdict_agent_id": (
                form.get("verdict_agent_id") or ""
            ).strip(),
        }

    def _csrf_guard(request: Request) -> None:
        """Reject cross-origin / cross-site state-mutating requests.

        Defense against the localhost-bound UI being POST'd by any tab the
        user has open on a malicious site (browsers send the request
        regardless of SameSite cookies on form-encoded POSTs). Two checks,
        either one of which passes is fine:

        * ``Sec-Fetch-Site: same-origin`` — modern browsers set this and it's
          the cleanest signal.
        * ``Origin`` header equals our own scheme://host — fallback for
          older browsers that don't send Sec-Fetch-Site.

        A request that fails both is rejected with 403. ``Origin`` MUST be
        present on cross-origin POSTs; same-origin POSTs from a browser
        page also send it. A direct ``curl`` without either header is
        treated as ambiguous and refused — the legitimate UI always has
        them. This is a localhost-only tool; we lose no real users.
        """
        sec_fetch_site = request.headers.get("sec-fetch-site", "").lower()
        if sec_fetch_site in ("same-origin", "none"):
            # "none" = top-level navigation typed into address bar (still
            # safe — the attacker page can't make this happen). "same-origin"
            # = the page making the request was loaded from us.
            return
        origin = request.headers.get("origin", "")
        if origin:
            host = request.headers.get("host", "")
            # Build the expected origin from the request's own host header,
            # not a hard-coded value. Localhost can be 127.0.0.1, localhost,
            # or an alias.
            expected = f"http://{host}"
            expected_https = f"https://{host}"
            if origin == expected or origin == expected_https:
                return
        raise HTTPException(
            status_code=403,
            detail=(
                "cross-origin request refused. This UI accepts POSTs only "
                "from its own pages."
            ),
        )

    async def _read_capped_form(request: Request) -> Any:
        """Read the form body, refusing anything over _CUSTOM_FORM_MAX_BYTES.

        Always enforces the CSRF guard first so a cross-origin attacker
        can't even reach the form parser. Then enforces the size cap by
        explicitly reading the body and counting bytes — Content-Length
        alone is forgeable (chunked transfer-encoding omits it entirely),
        so we don't trust it as the only line of defense.
        """
        _csrf_guard(request)
        # Reject when Content-Length is missing — closes the chunked-bypass
        # case. The legitimate editor form always sends Content-Length;
        # only an attacker hand-crafting a chunked POST omits it.
        cl = request.headers.get("content-length")
        if cl is None:
            raise HTTPException(
                status_code=411,
                detail="content-length header required",
            )
        try:
            cl_int = int(cl)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="invalid content-length"
            ) from exc
        if cl_int > _CUSTOM_FORM_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"form too large: max {_CUSTOM_FORM_MAX_BYTES} bytes",
            )
        # Re-verify after reading — Content-Length is advisory; the actual
        # body could be longer. Starlette buffers the whole body anyway
        # before form() returns, so this just makes the check load-bearing.
        body = await request.body()
        if len(body) > _CUSTOM_FORM_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"form too large: max {_CUSTOM_FORM_MAX_BYTES} bytes",
            )
        # Replay the buffered body so request.form() can parse it. Without
        # this, form() would try to read again from an exhausted stream.
        async def _replay() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = _replay  # type: ignore[attr-defined]
        return await request.form()

    @app.get("/custom-workflow/new", response_class=HTMLResponse)
    async def custom_workflow_new(request: Request) -> HTMLResponse:
        return _render_custom_editor(request, cw=None)

    @app.get("/custom-workflow/{workflow_id}", response_class=HTMLResponse)
    async def custom_workflow_edit(
        request: Request, workflow_id: str
    ) -> HTMLResponse:
        cw = load_custom_workflow(workflow_id)
        if cw is None:
            raise HTTPException(status_code=404, detail="custom workflow not found")
        return _render_custom_editor(request, cw=cw)

    @app.post("/custom-workflow", response_model=None)
    async def custom_workflow_create(request: Request):  # noqa: ANN202
        form = await _read_capped_form(request)
        values = _form_values_from_post(form)
        try:
            cw = validate_custom_workflow_form(
                label=values["label"],
                domain_noun=values["domain_noun"],
                decision_verb=values["decision_verb"],
                agent_ids=[aid for aid, _ in values["agents"]],
                agent_models=[mdl for _, mdl in values["agents"]],
                verdict_agent_id=values["verdict_agent_id"],
                rubric_paragraph=values["rubric_paragraph"],
                workflow_id=None,
            )
        except CustomFormError as exc:
            return _render_custom_editor(
                request,
                cw=None,
                form_errors=exc.errors,
                form_values=values,
                status_code=400,
            )
        try:
            wf = save_custom_workflow(cw)
        except (OSError, ValueError) as exc:
            # Narrow except: OSError covers disk full / permission /
            # symlink issues; ValueError covers schema invariant misses
            # (e.g. verdict-not-last). A wider Exception would hide real
            # bugs. Log server-side, show a generic message to the user.
            _log.warning(
                "custom_workflow_create save failed: %s", exc, exc_info=True
            )
            return _render_custom_editor(
                request,
                cw=None,
                form_errors={
                    "_general": (
                        "Could not save the workflow. Check the server log "
                        "for details."
                    )
                },
                form_values=values,
                status_code=500,
            )
        return RedirectResponse(
            url=f"/setup?workflow={wf.key}", status_code=303
        )

    @app.post("/custom-workflow/{workflow_id}", response_model=None)
    async def custom_workflow_update(
        request: Request, workflow_id: str
    ):  # noqa: ANN202
        existing = load_custom_workflow(workflow_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="custom workflow not found")
        form = await _read_capped_form(request)
        values = _form_values_from_post(form)
        try:
            cw = validate_custom_workflow_form(
                label=values["label"],
                domain_noun=values["domain_noun"],
                decision_verb=values["decision_verb"],
                agent_ids=[aid for aid, _ in values["agents"]],
                agent_models=[mdl for _, mdl in values["agents"]],
                verdict_agent_id=values["verdict_agent_id"],
                rubric_paragraph=values["rubric_paragraph"],
                workflow_id=workflow_id,  # preserve id on edit
            )
        except CustomFormError as exc:
            return _render_custom_editor(
                request,
                cw=existing,
                form_errors=exc.errors,
                form_values=values,
                status_code=400,
            )
        try:
            wf = save_custom_workflow(cw)
        except (OSError, ValueError) as exc:
            _log.warning(
                "custom_workflow_update save failed: %s", exc, exc_info=True
            )
            return _render_custom_editor(
                request,
                cw=existing,
                form_errors={
                    "_general": (
                        "Could not save the workflow. Check the server log "
                        "for details."
                    )
                },
                form_values=values,
                status_code=500,
            )
        return RedirectResponse(
            url=f"/setup?workflow={wf.key}", status_code=303
        )

    @app.post("/custom-workflow/{workflow_id}/delete", response_model=None)
    async def custom_workflow_delete(
        request: Request, workflow_id: str
    ):  # noqa: ANN202
        _csrf_guard(request)
        # Validate the id format before touching disk — refuses any
        # non-conforming string at the path level.
        from topofuzz.ui.custom_storage import is_valid_workflow_id

        if not is_valid_workflow_id(workflow_id):
            raise HTTPException(status_code=400, detail="invalid workflow id")
        delete_custom_workflow(workflow_id)
        return RedirectResponse(url="/setup", status_code=303)

    def _resolve_target_dir(workflow_id: str) -> Path:
        """Resolve the on-disk dir for a workflow id, with containment check."""
        from topofuzz.ui.custom_storage import workflow_dir

        return workflow_dir(workflow_id)

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
