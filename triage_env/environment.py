"""
SupportTriageEnv — OpenEnv compliant Customer Support Triage environment.

API:
    env = SupportTriageEnv()
    obs = env.reset("task1-easy")
    result = env.step(action)
    state = env.state()
    summary = env.episode_summary()
"""

from __future__ import annotations
import time
import copy
from typing import Any, Dict, List, Optional

from triage_env.models import (
    Action, ActionType, Observation, Reward, StepResult,
    EpisodeSummary, Ticket, TicketCategory, TicketPriority,
)
from triage_env.tickets import TASKS, GROUND_TRUTH
from triage_env.reward import compute_reward
from graders.graders import GRADERS


class SupportTriageEnv:
    """
    OpenEnv-compliant environment for customer support ticket triage.

    Tasks:
        task1-easy    — 5 tickets, classify + prioritize
        task2-medium  — 8 tickets, classify + prioritize + route
        task3-hard    — 12 tickets, classify + prioritize + route + reply
    """

    VERSION = "1.0.0"
    ENV_ID  = "support-triage-v1"

    def __init__(self):
        self._task_id:      Optional[str]     = None
        self._task_config:  Optional[Dict]    = None
        self._tickets:      List[Ticket]      = []
        self._cursor:       int               = 0
        self._step_count:   int               = 0
        self._actions_log:  List[Dict]        = []
        self._rewards_log:  List[float]       = []
        self._start_time:   float             = 0.0
        self._last_action_time: float         = 0.0
        self._episode_reward: float           = 0.0

    # ─── OpenEnv Core API ────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1-easy") -> Observation:
        """Start a new episode. Returns initial observation."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}")

        self._task_id     = task_id
        self._task_config = TASKS[task_id]
        self._tickets     = copy.deepcopy(self._task_config["tickets"])
        self._cursor      = 0
        self._step_count  = 0
        self._actions_log = []
        self._rewards_log = []
        self._episode_reward = 0.0
        self._start_time  = time.time()
        self._last_action_time = time.time()

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one action on the current ticket.
        Returns (observation, reward, done, info).
        """
        if self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._cursor >= len(self._tickets):
            raise RuntimeError("Episode is already done. Call reset().")

        elapsed = time.time() - self._last_action_time
        self._last_action_time = time.time()

        current_ticket = self._tickets[self._cursor]
        gt = GROUND_TRUTH.get(current_ticket.id, {})

        # Evaluate reply quality if provided
        reply_quality = 0.0
        if action.reply_text and self._task_config.get("requires_reply"):
            reply_quality = self._heuristic_reply_quality(
                action.reply_text, current_ticket.id
            )

        reward = compute_reward(
            action=action,
            ground_truth=gt,
            task_config=self._task_config,
            elapsed_seconds=elapsed,
            reply_quality_score=reply_quality,
        )

        # Log action for grader
        self._actions_log.append({
            "ticket_id":  current_ticket.id,
            "action_type": action.action_type,
            "category":   action.category,
            "priority":   action.priority,
            "team":       action.team,
            "reply_text": action.reply_text,
            "reasoning":  action.reasoning,
            "reward":     reward.total,
            "step":       self._step_count,
        })
        self._rewards_log.append(reward.total)
        self._episode_reward += reward.total
        self._step_count += 1

        # Advance cursor
        self._cursor += 1
        done = self._cursor >= len(self._tickets)

        if not done:
            next_obs = self._make_observation()
        else:
            # Final observation (empty)
            next_obs = Observation(
                step=self._step_count,
                ticket=current_ticket,   # last ticket seen
                inbox_remaining=0,
                inbox_total=len(self._tickets),
                actions_taken=self._actions_log,
                episode_reward_so_far=self._episode_reward,
                task_id=self._task_id,
            )

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "ticket_id": current_ticket.id,
                "step": self._step_count,
                "episode_reward": self._episode_reward,
                "done": done,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Returns full mutable environment state (for debugging/logging)."""
        return {
            "env_id":          self.ENV_ID,
            "version":         self.VERSION,
            "task_id":         self._task_id,
            "step":            self._step_count,
            "cursor":          self._cursor,
            "total_tickets":   len(self._tickets),
            "episode_reward":  self._episode_reward,
            "actions_log":     self._actions_log,
            "rewards_log":     self._rewards_log,
            "elapsed_seconds": time.time() - self._start_time if self._start_time else 0,
        }

    # ─── Episode Summary & Grading ───────────────────────────────────────────

    def episode_summary(self) -> EpisodeSummary:
        """Run the task grader and return normalised episode results."""
        if not self._task_id:
            raise RuntimeError("No episode has been run.")

        grader = GRADERS[self._task_id]
        ticket_ids = [a["ticket_id"] for a in self._actions_log]
        result = grader.grade(self._actions_log, ticket_ids)

        n = len(self._tickets)
        cat_correct  = sum(1 for a in self._actions_log
                          if GROUND_TRUTH.get(a["ticket_id"], {}).get("category") == a.get("category"))
        pri_correct  = sum(1 for a in self._actions_log
                          if GROUND_TRUTH.get(a["ticket_id"], {}).get("priority") == a.get("priority"))
        route_correct = sum(1 for a in self._actions_log
                           if GROUND_TRUTH.get(a["ticket_id"], {}).get("team") == a.get("team"))

        max_reward = (
            self._task_config["max_reward_per_ticket"] * len(self._tickets)
            if self._task_config else 1.0
        )

        return EpisodeSummary(
            task_id=self._task_id,
            total_reward=round(self._episode_reward, 4),
            max_possible_reward=max_reward,
            score=result.score,
            steps=self._step_count,
            tickets_processed=len(self._actions_log),
            category_accuracy=round(cat_correct / n, 4) if n else 0.0,
            priority_accuracy=round(pri_correct / n, 4) if n else 0.0,
            routing_accuracy=round(route_correct / n, 4) if n else 0.0,
            duration_seconds=round(time.time() - self._start_time, 2),
            per_ticket_rewards=self._rewards_log,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        ticket = self._tickets[self._cursor]
        hints  = self._get_hints(ticket)
        return Observation(
            step=self._step_count,
            ticket=ticket,
            inbox_remaining=len(self._tickets) - self._cursor,
            inbox_total=len(self._tickets),
            actions_taken=self._actions_log[-3:],   # last 3 for context
            episode_reward_so_far=self._episode_reward,
            task_id=self._task_id,
            hints=hints,
        )

    def _get_hints(self, ticket: Ticket) -> List[str]:
        """Optional coaching hints injected into observation."""
        hints = []
        task_cfg = self._task_config or {}
        if task_cfg.get("requires_routing"):
            hints.append("This task requires routing to the correct team.")
        if task_cfg.get("requires_reply"):
            hints.append("Draft a reply for tickets that need a response.")
        if ticket.metadata.get("requires_reply") is False:
            hints.append("No reply needed for this ticket type.")
        return hints

    def _heuristic_reply_quality(self, reply: str, ticket_id: str) -> float:
        """Fast heuristic reply quality — no external API needed."""
        if not reply or len(reply.strip()) < 20:
            return 0.0
        score = 0.4
        words = len(reply.split())
        if words >= 40: score += 0.2
        if words >= 80: score += 0.1
        bad = ["placeholder", "[insert", "xxx", "todo"]
        if any(b in reply.lower() for b in bad):
            score -= 0.3
        gt_pri = GROUND_TRUTH.get(ticket_id, {}).get("priority")
        if gt_pri in (TicketPriority.CRITICAL, TicketPriority.HIGH):
            empathy = ["sorry", "apologize", "immediately", "urgent", "right away"]
            if any(e in reply.lower() for e in empathy):
                score += 0.2
        return max(0.0, min(1.0, score))

    # ─── Convenience ─────────────────────────────────────────────────────────

    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "fields": {
                "action_type": [a.value for a in ActionType],
                "category":    [c.value for c in TicketCategory],
                "priority":    [p.value for p in TicketPriority],
                "team":        "string",
                "reply_text":  "string (optional)",
                "reasoning":   "string (optional)",
            }
        }

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "fields": {
                "step":                   "int",
                "ticket":                 "Ticket object (id, subject, body, sender_email, ...)",
                "inbox_remaining":        "int",
                "inbox_total":            "int",
                "actions_taken":          "list[dict] — last 3 actions",
                "episode_reward_so_far":  "float",
                "task_id":                "string",
                "hints":                  "list[str]",
            }
        }

    def available_teams(self) -> List[str]:
        return [
            "billing-team", "tech-team", "account-team", "shipping-team",
            "returns-team", "support-team", "security-team", "legal-team",
            "product-team", "sales-team", "partnerships-team", "spam-filter",
        ]
