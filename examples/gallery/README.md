# topofuzz gallery

Worked LangGraph fixtures you can run with `topofuzz`. Each directory
is a self-contained example: `fixture.py`, eval CSV, and (where
needed) a rubric.

These are small illustrative topologies, not a benchmark. Numbers from
any local run will depend on your models, keys, and seed.

| Slug | What it is |
|------|------------|
| [`langgraph-lending-4role/`](langgraph-lending-4role/) | 4-agent loan underwriting chain |
| [`langgraph-adversarial-relay/`](langgraph-adversarial-relay/) | 4-agent skeptic-in-the-middle loan chain |
| [`insurance-claim-triage/`](insurance-claim-triage/) | Health-insurance claims triage |
| [`langgraph-content-moderation/`](langgraph-content-moderation/) | UGC moderation chain |
| [`langgraph-supervisor-research/`](langgraph-supervisor-research/) | Supervisor + workers |
| [`langgraph-hiring-4role/`](langgraph-hiring-4role/) | Hiring decision chain |
| [`langgraph-insurance-4role/`](langgraph-insurance-4role/) | Life-insurance underwriting |
| [`langgraph-admissions-4role/`](langgraph-admissions-4role/) | University admissions |
| [`langgraph-benefits-4role/`](langgraph-benefits-4role/) | Public-benefits eligibility |

## Run one

Bring your own vendor key (`GEMINI_API_KEY`, or another LiteLLM-supported
provider). Then:

```powershell
topofuzz --accept-license run `
  examples/gallery/langgraph-lending-4role/fixture.py:build_graph `
  --eval examples/gallery/langgraph-lending-4role/eval.csv `
  --n 8 --variants 4 --out runs --yes
```

Use `--simulate` first if you only want to check wiring (no API calls).
