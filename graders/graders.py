"""
Agent graders — programmatic scoring per task.
Each grader receives the full episode history and returns a score 0.0–1.0.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from triage_env.models import Action, TicketCategory, TicketPriority, EpisodeSummary
from triage_env.tickets import GROUND_TRUTH


@dataclass
class GraderResult:
    score: float               # 0.0 – 1.0
    label: str                 # "pass" / "partial" / "fail"
    details: Dict[str, Any]


def _label(score: float) -> str:
    if score >= 0.85:
        return "pass"
    if score >= 0.5:
        return "partial"
    return "fail"


# ─── Task 1 Grader ────────────────────────────────────────────────────────────

class Task1Grader:
    """
    Easy: purely category + priority accuracy.
    Pass threshold: >= 80% on both metrics.
    """
    name = "task1-easy"
    description = "Basic triage: category + priority accuracy on 5 clear tickets."

    def grade(self, actions: List[Dict[str, Any]], ticket_ids: List[str]) -> GraderResult:
        cat_correct = 0
        pri_correct = 0
        n = len(ticket_ids)

        details = []
        for action_dict, tid in zip(actions, ticket_ids):
            gt = GROUND_TRUTH.get(tid, {})
            pred_cat = action_dict.get("category")
            pred_pri = action_dict.get("priority")

            c_ok = pred_cat == gt.get("category")
            p_ok = pred_pri == gt.get("priority")
            cat_correct += int(c_ok)
            pri_correct += int(p_ok)

            details.append({
                "ticket_id": tid,
                "category_correct": c_ok,
                "priority_correct": p_ok,
                "predicted_category": pred_cat,
                "true_category": gt.get("category"),
                "predicted_priority": pred_pri,
                "true_priority": gt.get("priority"),
            })

        cat_acc = cat_correct / n if n > 0 else 0.0
        pri_acc = pri_correct / n if n > 0 else 0.0
        score = (cat_acc * 0.6 + pri_acc * 0.4)
        score = max(0.0001, min(0.9999, score))

        return GraderResult(
            score=round(score, 4),
            label=_label(score),
            details={
                "category_accuracy": round(cat_acc, 4),
                "priority_accuracy": round(pri_acc, 4),
                "per_ticket": details,
            },
        )


# ─── Task 2 Grader ────────────────────────────────────────────────────────────

class Task2Grader:
    """
    Medium: category + priority + routing accuracy on 8 ambiguous tickets.
    Partial credit for adjacent priority mistakes.
    Pass threshold: >= 70% weighted score.
    """
    name = "task2-medium"
    description = "Ambiguous triage + routing: 8 tickets with partial credit."

    PRIORITY_ORDER = {
        TicketPriority.CRITICAL: 4,
        TicketPriority.HIGH:     3,
        TicketPriority.MEDIUM:   2,
        TicketPriority.LOW:      1,
        TicketPriority.IGNORE:   0,
    }

    def _priority_score(self, pred, true) -> float:
        if pred == true:
            return 1.0
        if pred is None or true is None:
            return 0.0
        diff = abs(self.PRIORITY_ORDER.get(pred, 0) - self.PRIORITY_ORDER.get(true, 0))
        return max(0.0, 1.0 - diff * 0.4)

    def grade(self, actions: List[Dict[str, Any]], ticket_ids: List[str]) -> GraderResult:
        total_cat = total_pri = total_route = 0.0
        n = len(ticket_ids)
        details = []

        for action_dict, tid in zip(actions, ticket_ids):
            gt = GROUND_TRUTH.get(tid, {})
            pred_cat   = action_dict.get("category")
            pred_pri   = action_dict.get("priority")
            pred_team  = action_dict.get("team")

            c_ok    = float(pred_cat == gt.get("category"))
            p_score = self._priority_score(pred_pri, gt.get("priority"))
            r_ok    = float(pred_team == gt.get("team"))

            total_cat   += c_ok
            total_pri   += p_score
            total_route += r_ok

            details.append({
                "ticket_id": tid,
                "category_score": c_ok,
                "priority_score": p_score,
                "routing_score": r_ok,
            })

        cat_acc   = total_cat   / n if n > 0 else 0.0
        pri_acc   = total_pri   / n if n > 0 else 0.0
        route_acc = total_route / n if n > 0 else 0.0

        # Weighted: category most important, then priority, then routing
        score = cat_acc * 0.45 + pri_acc * 0.30 + route_acc * 0.25
        score = max(0.0001, min(0.9999, score))

        return GraderResult(
            score=round(score, 4),
            label=_label(score),
            details={
                "category_accuracy": round(cat_acc, 4),
                "priority_accuracy": round(pri_acc, 4),
                "routing_accuracy": round(route_acc, 4),
                "per_ticket": details,
            },
        )


# ─── Task 3 Grader ────────────────────────────────────────────────────────────

class Task3Grader:
    """
    Hard: full pipeline grader.
    Category + priority + routing + reply quality (heuristic).
    Pass threshold: >= 65% weighted score.
    Reply quality graded by heuristic rules (length, tone keywords, personalisation).
    """
    name = "task3-hard"
    description = "Full pipeline: 12 complex tickets, includes reply quality scoring."

    PRIORITY_ORDER = {
        TicketPriority.CRITICAL: 4,
        TicketPriority.HIGH:     3,
        TicketPriority.MEDIUM:   2,
        TicketPriority.LOW:      1,
        TicketPriority.IGNORE:   0,
    }

    def _priority_score(self, pred, true) -> float:
        if pred == true:
            return 1.0
        if pred is None or true is None:
            return 0.0
        diff = abs(self.PRIORITY_ORDER.get(pred, 0) - self.PRIORITY_ORDER.get(true, 0))
        return max(0.0, 1.0 - diff * 0.4)

    def _reply_quality(self, reply: Optional[str], ticket_id: str) -> float:
        """
        Heuristic reply quality scorer (no LLM dependency for reproducibility).
        Checks: non-empty, minimum length, contains apology/acknowledgement for
        high-severity tickets, doesn't contain placeholder text.
        """
        from triage_env.tickets import TASK3_TICKETS
        ticket_meta = next((t.metadata for t in TASK3_TICKETS if t.id == ticket_id), {})

        if not reply or len(reply.strip()) < 20:
            return 0.0

        score = 0.3  # baseline for any non-empty reply

        # Length signal: good replies are substantive
        words = len(reply.split())
        if words >= 30:
            score += 0.2
        if words >= 60:
            score += 0.1

        # Check for placeholder text (bad)
        bad_markers = ["[your name]", "[insert", "xxx", "todo", "placeholder"]
        if any(m in reply.lower() for m in bad_markers):
            score -= 0.2

        # Empathy markers for sensitive tickets
        gt_pri = GROUND_TRUTH.get(ticket_id, {}).get("priority")
        if gt_pri in (TicketPriority.CRITICAL, TicketPriority.HIGH):
            empathy_words = ["sorry", "apologize", "understand", "urgent", "immediately",
                             "right away", "escalat", "priority", "sincerely"]
            if any(w in reply.lower() for w in empathy_words):
                score += 0.2

        # Tone appropriateness for spam/ignore tickets: should NOT have a reply
        if ticket_meta.get("requires_reply") is False and reply and len(reply) > 10:
            score -= 0.3   # penalize replying to spam

        return max(0.0, min(1.0, round(score, 3)))

    def grade(self, actions: List[Dict[str, Any]], ticket_ids: List[str]) -> GraderResult:
        total_cat = total_pri = total_route = total_reply = 0.0
        reply_count = 0
        n = len(ticket_ids)
        details = []

        for action_dict, tid in zip(actions, ticket_ids):
            gt = GROUND_TRUTH.get(tid, {})
            pred_cat   = action_dict.get("category")
            pred_pri   = action_dict.get("priority")
            pred_team  = action_dict.get("team")
            pred_reply = action_dict.get("reply_text")

            c_ok    = float(pred_cat == gt.get("category"))
            p_score = self._priority_score(pred_pri, gt.get("priority"))
            r_ok    = float(pred_team == gt.get("team"))
            rq      = self._reply_quality(pred_reply, tid)

            total_cat   += c_ok
            total_pri   += p_score
            total_route += r_ok
            total_reply += rq
            reply_count += 1

            details.append({
                "ticket_id": tid,
                "category_score": c_ok,
                "priority_score": p_score,
                "routing_score": r_ok,
                "reply_quality": rq,
            })

        cat_acc   = total_cat   / n if n > 0 else 0.0
        pri_acc   = total_pri   / n if n > 0 else 0.0
        route_acc = total_route / n if n > 0 else 0.0
        reply_avg = total_reply / reply_count if reply_count > 0 else 0.0

        score = (cat_acc * 0.30 + pri_acc * 0.20 +
                 route_acc * 0.20 + reply_avg * 0.30)
        score = max(0.0001, min(0.9999, score))

        return GraderResult(
            score=round(score, 4),
            label=_label(score),
            details={
                "category_accuracy":  round(cat_acc, 4),
                "priority_accuracy":  round(pri_acc, 4),
                "routing_accuracy":   round(route_acc, 4),
                "reply_quality_avg":  round(reply_avg, 4),
                "per_ticket": details,
            },
        )


# ─── Registry ─────────────────────────────────────────────────────────────────

GRADERS = {
    "task1-easy":   Task1Grader(),
    "task2-medium": Task2Grader(),
    "task3-hard":   Task3Grader(),
}
