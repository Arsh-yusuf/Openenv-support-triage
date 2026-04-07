"""
OpenEnv Typed Models — Customer Support Ticket Triage
All Observation, Action, and Reward types are fully typed Pydantic models.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ─── Enumerations ────────────────────────────────────────────────────────────

class TicketCategory(str, Enum):
    BILLING        = "billing"
    TECHNICAL      = "technical"
    ACCOUNT        = "account"
    SHIPPING       = "shipping"
    RETURNS        = "returns"
    GENERAL        = "general"
    SPAM           = "spam"

class TicketPriority(str, Enum):
    CRITICAL = "critical"   # SLA breach, data loss, outage
    HIGH     = "high"       # Major blocker, angry customer
    MEDIUM   = "medium"     # Normal issue, needs attention
    LOW      = "low"        # Minor, can wait
    IGNORE   = "ignore"     # Spam / duplicate / no action needed

class ActionType(str, Enum):
    CLASSIFY  = "classify"   # Set category + priority
    ROUTE     = "route"      # Assign to a team/agent
    REPLY     = "reply"      # Draft a reply
    ESCALATE  = "escalate"   # Escalate to supervisor
    CLOSE     = "close"      # Mark resolved
    SKIP      = "skip"       # Move to next ticket (penalty)


# ─── Ticket (world object) ────────────────────────────────────────────────────

class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    sender_email: str
    received_at: str                   # ISO8601
    attachments: List[str] = []
    metadata: Dict[str, Any] = {}

    # Ground truth (hidden from agent during episode)
    _true_category: Optional[TicketCategory] = None
    _true_priority:  Optional[TicketPriority]  = None
    _expected_team:  Optional[str]             = None


# ─── Observation ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    step: int
    ticket: Ticket
    inbox_remaining: int
    inbox_total: int
    actions_taken: List[Dict[str, Any]] = []
    episode_reward_so_far: float = 0.0
    task_id: str
    hints: List[str] = []              # Optional coaching hints


# ─── Action ──────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """What the agent sends back."""
    action_type: ActionType
    category:    Optional[TicketCategory] = None
    priority:    Optional[TicketPriority]  = None
    team:        Optional[str]             = None   # e.g. "billing-team"
    reply_text:  Optional[str]             = None
    reasoning:   Optional[str]             = None   # CoT explanation


# ─── Reward ──────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Shaped reward with breakdown for transparency."""
    total: float = Field(..., ge=-1.0, le=1.0)

    category_correct:  Optional[bool]  = None
    priority_correct:  Optional[bool]  = None
    routing_correct:   Optional[bool]  = None
    reply_quality:     Optional[float] = None   # 0.0–1.0 from LLM grader
    escalation_apt:    Optional[bool]  = None
    speed_bonus:       float = 0.0
    penalty:           float = 0.0
    breakdown:         Dict[str, float] = {}


# ─── Step Result ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


# ─── Episode Summary ─────────────────────────────────────────────────────────

class EpisodeSummary(BaseModel):
    task_id: str
    total_reward: float
    max_possible_reward: float
    score: float                  # 0.0–1.0 normalised
    steps: int
    tickets_processed: int
    category_accuracy: float
    priority_accuracy: float
    routing_accuracy: float
    duration_seconds: float
    per_ticket_rewards: List[float] = []
