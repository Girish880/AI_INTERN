# backend/main.py
import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv  # ✅ added

# Load environment variables (e.g., OPENAI_API_KEY from .env)
load_dotenv()

# Import agents (if available, otherwise placeholders)
try:
    from agents.planner_agent import PlannerAgent
    from agents.ranker_agent import RankerAgent
    from agents.orchestrator_agent import OrchestratorAgent
    from agents.analyzer_agent import AnalyzerAgent
except Exception:
    PlannerAgent = None
    RankerAgent = None
    OrchestratorAgent = None
    AnalyzerAgent = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-agent-game-tester")

app = FastAPI(title="Multi-Agent Game Tester POC")

# Allow local frontends to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request / Response models
# ----------------------------
class PlanRequest(BaseModel):
    target_url: str
    seeds: Optional[List[str]] = None
    n_candidates: Optional[int] = 20


class PlanResponse(BaseModel):
    candidates: List[Dict[str, Any]]


class RankRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    top_k: Optional[int] = 10


class RankResponse(BaseModel):
    top_candidates: List[Dict[str, Any]]


class ExecuteRequest(BaseModel):
    tests: List[Dict[str, Any]]
    parallelism: Optional[int] = 3


class ExecuteResponse(BaseModel):
    run_id: str
    results: List[Dict[str, Any]]


class AnalyzeRequest(BaseModel):
    run_id: str
    results: List[Dict[str, Any]]


class AnalyzeResponse(BaseModel):
    report_path: str
    report: Dict[str, Any]


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Game Tester API is up. Use /plan, /rank, /execute, /analyze, /report/{run_id}"
    }


@app.post("/plan", response_model=PlanResponse)
async def plan(request: PlanRequest):
    """
    Generate candidate test cases using PlannerAgent (LangChain).
    """
    logger.info("Received plan request: %s", request.target_url)
    if PlannerAgent is None:
        # placeholder candidate generation
        candidates = [
            {
                "id": f"cand_{i+1}",
                "description": f"Auto-generated test #{i+1}",
                "seed": request.seeds or [],
            }
            for i in range(request.n_candidates or 20)
        ]
    else:
        planner = PlannerAgent()
        candidates = await planner.generate(
            request.target_url, seeds=request.seeds, n=request.n_candidates
        )

    return {"candidates": candidates}


@app.post("/rank", response_model=RankResponse)
async def rank(request: RankRequest):
    """
    Rank test candidates and return top_k.
    """
    logger.info(
        "Ranking %d candidates; return top %d", len(request.candidates), request.top_k
    )
    if RankerAgent is None:
        # naive placeholder: return the first top_k
        top = request.candidates[: request.top_k]
    else:
        ranker = RankerAgent()
        top = await ranker.rank_and_select(request.candidates, top_k=request.top_k)
    return {"top_candidates": top}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """
    Orchestrator coordinates execution across ExecutorAgents.
    Returns a run_id and immediate results (artifacts saved on disk).
    """
    logger.info(
        "Starting execution for %d tests with parallelism=%s",
        len(request.tests),
        request.parallelism,
    )
    run_id = f"run_{os.urandom(4).hex()}"

    if OrchestratorAgent is None:
        # placeholder: simulate execution results
        results = []
        for t in request.tests:
            results.append(
                {
                    "test_id": t.get("id"),
                    "status": "completed",
                    "verdict": "unknown",  # analyzer will determine
                    "artifacts": {
                        "screenshots": [],
                        "dom_snapshot": None,
                        "console_logs": [],
                        "network_log": None,
                    },
                }
            )
    else:
        orchestrator = OrchestratorAgent()
        results = await orchestrator.execute_tests(
            run_id, request.tests, parallelism=request.parallelism
        )

    # Save raw results for debugging / triage
    os.makedirs("reports", exist_ok=True)
    raw_path = f"reports/{run_id}_raw.json"
    import json

    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return {"run_id": run_id, "results": results}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    AnalyzerAgent validates runs and writes a JSON report.
    """
    logger.info(
        "Analyzing run %s with %d results", request.run_id, len(request.results)
    )
    if AnalyzerAgent is None:
        # placeholder analysis
        report = {
            "run_id": request.run_id,
            "summary": {
                "total": len(request.results),
                "passed": 0,
                "failed": 0,
                "flaky": len(request.results),
            },
            "tests": [
                {
                    "test_id": r.get("test_id"),
                    "verdict": "flaky",
                    "artifacts": r.get("artifacts", {}),
                    "reproducibility": {"repeats": 1, "stable": False},
                    "notes": "Placeholder analysis. Implement AnalyzerAgent for real validation.",
                }
                for r in request.results
            ],
        }
        report_path = f"reports/{request.run_id}_report.json"
        os.makedirs("reports", exist_ok=True)
        import json

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    else:
        analyzer = AnalyzerAgent()
        report_path, report = await analyzer.analyze_and_write_report(
            request.run_id, request.results
        )

    return {"report_path": report_path, "report": report}


@app.get("/report/{run_id}")
async def get_report(run_id: str):
    """
    Return the generated report JSON for a run id if exists.
    """
    import json

    path = f"reports/{run_id}_report.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
