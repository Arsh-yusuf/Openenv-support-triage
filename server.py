"""
OpenEnv HTTP API — Customer Support Triage Environment
Exposes step / reset / state endpoints over FastAPI.
"""

from __future__ import annotations
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os

from triage_env.environment import SupportTriageEnv
from triage_env.models import Action, ActionType, TicketCategory, TicketPriority

app = FastAPI(
    title="OpenEnv: Customer Support Triage",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage, "
        "route, and respond to customer support tickets."
    ),
    version="1.0.0",
    docs_url="/docs",
)

# Mount static files for the demo UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# One global env instance per server (stateful demo)
_env = SupportTriageEnv()
_current_obs = None


# ─── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1-easy"

class StepRequest(BaseModel):
    action_type: str
    category:   Optional[str] = None
    priority:   Optional[str] = None
    team:       Optional[str] = None
    reply_text: Optional[str] = None
    reasoning:  Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        with open(index) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>OpenEnv: Support Triage</h1><p>Visit /docs</p>")


@app.post("/reset")
async def reset(req: ResetRequest):
    global _current_obs
    try:
        obs = _env.reset(req.task_id)
        _current_obs = obs
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    global _current_obs
    if _env._task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _env._cursor >= len(_env._tickets):
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")

    try:
        action = Action(
            action_type=ActionType(req.action_type),
            category=TicketCategory(req.category)   if req.category  else None,
            priority=TicketPriority(req.priority)   if req.priority  else None,
            team=req.team,
            reply_text=req.reply_text,
            reasoning=req.reasoning,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    result = _env.step(action)
    _current_obs = result.observation

    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward.model_dump(),
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state")
async def state():
    return _env.state()


@app.get("/summary")
async def summary():
    if not _env._actions_log:
        raise HTTPException(status_code=400, detail="No episode data yet.")
    return _env.episode_summary().model_dump()


@app.get("/info")
async def info():
    return {
        "env_id":      SupportTriageEnv.ENV_ID,
        "version":     SupportTriageEnv.VERSION,
        "tasks":       ["task1-easy", "task2-medium", "task3-hard"],
        "action_space":      _env.action_space,
        "observation_space": _env.observation_space,
        "available_teams":   _env.available_teams(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "env": SupportTriageEnv.ENV_ID}


if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=7860, reload=False)
