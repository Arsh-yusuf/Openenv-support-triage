"""
Reward shaping for OpenEnv Customer Support Triage.

Philosophy:
- Category correct:  +0.4  (most important signal)
- Priority correct:  +0.3  (second most important)
- Routing correct:   +0.2  (medium; only in tasks 2+)
- Reply quality:     +0.5  (LLM-judged; only in task 3)
- Speed bonus:       +0.1  (if action taken in < 5 seconds)
- Skip penalty:     -0.3   (penalize skipping a ticket)
- Wrong escalation: -0.2   (escalating low-priority wastes resources)
- Spam classified:  +0.4   (bonus for catching spam without false positives)
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, Tuple
from triage_env.models import (
    Action, ActionType, Reward,
    TicketCategory, TicketPriority,
)

# Priority adjacency — adjacent misses are penalised less than complete misses
PRIORITY_ADJACENCY: Dict[TicketPriority, int] = {
    TicketPriority.CRITICAL: 4,
    TicketPriority.HIGH:     3,
    TicketPriority.MEDIUM:   2,
    TicketPriority.LOW:      1,
    TicketPriority.IGNORE:   0,
}


def priority_partial_score(pred: TicketPriority, true: TicketPriority) -> float:
    """Returns 1.0 for exact match, 0.5 for one step off, 0.0 for large miss."""
    if pred == true:
        return 1.0
    diff = abs(PRIORITY_ADJACENCY[pred] - PRIORITY_ADJACENCY[true])
    if diff == 1:
        return 0.5
    return 0.0


def compute_reward(
    action: Action,
    ground_truth: Dict[str, Any],
    task_config: Dict[str, Any],
    elapsed_seconds: float = 999.0,
    reply_quality_score: float = 0.0,
) -> Reward:
    """
    Compute shaped reward for a single action on a single ticket.
    Returns a Reward object with full breakdown.
    """
    breakdown: Dict[str, float] = {}
    total = 0.0

    true_category = ground_truth["category"]
    true_priority  = ground_truth["priority"]
    true_team      = ground_truth["team"]

    # ── Skip penalty ──────────────────────────────────────────────────────
    if action.action_type == ActionType.SKIP:
        breakdown["skip_penalty"] = 0.01
        return Reward(
            total=0.01,
            penalty=0.01,
            breakdown=breakdown,
        )

    # ── Category scoring ──────────────────────────────────────────────────
    category_correct = action.category == true_category
    if category_correct:
        score = 0.4
        # Extra bonus for correctly catching spam (reduces false positive risk)
        if true_category == TicketCategory.SPAM:
            score += 0.1
        breakdown["category_correct"] = score
        total += score
    else:
        breakdown["category_wrong"] = 0.0

    # ── Priority scoring ──────────────────────────────────────────────────
    if action.priority is not None:
        pri_score = priority_partial_score(action.priority, true_priority) * 0.3
        breakdown["priority_score"] = pri_score
        total += pri_score
        priority_correct = action.priority == true_priority
    else:
        priority_correct = False
        breakdown["priority_score"] = 0.0

    # ── Routing scoring (tasks 2+) ────────────────────────────────────────
    routing_correct = False
    if task_config.get("requires_routing") and action.team:
        if action.team == true_team:
            routing_correct = True
            breakdown["routing_correct"] = 0.2
            total += 0.2
        else:
            # Partial credit: at least correct department family?
            pred_family = action.team.split("-")[0] if action.team else ""
            true_family = true_team.split("-")[0] if true_team else ""
            if pred_family == true_family:
                breakdown["routing_partial"] = 0.1
                total += 0.1
            else:
                breakdown["routing_wrong"] = 0.0

    # ── Reply quality (task 3) ────────────────────────────────────────────
    if task_config.get("requires_reply") and action.reply_text:
        rq = reply_quality_score * 0.5
        breakdown["reply_quality"] = rq
        total += rq

    # ── Escalation check ─────────────────────────────────────────────────
    if action.action_type == ActionType.ESCALATE:
        if true_priority in (TicketPriority.CRITICAL, TicketPriority.HIGH):
            breakdown["escalation_apt"] = 0.05
            total += 0.05
        else:
            # Escalating a low-priority ticket wastes supervisor time
            breakdown["escalation_penalty"] = -0.2
            total -= 0.2

    # ── Speed bonus ───────────────────────────────────────────────────────
    speed_bonus = 0.0
    if elapsed_seconds < 5.0:
        speed_bonus = 0.1
        breakdown["speed_bonus"] = speed_bonus
        total += speed_bonus

    # Clamp to strictly (0, 1) — Scaler validator requires values exclusively between 0 and 1
    total = max(0.01, min(0.99, total))

    return Reward(
        total=total,
        category_correct=category_correct,
        priority_correct=priority_correct,
        routing_correct=routing_correct,
        reply_quality=reply_quality_score if task_config.get("requires_reply") else None,
        speed_bonus=speed_bonus,
        breakdown=breakdown,
    )
