#!/usr/bin/env python3
"""
Baseline inference script — runs a Open AI model against all 3 tasks.

Usage:
    export HF_TOKEN=hf_...
    python baseline/run_baseline.py [--task all]

Produces reproducible baseline scores for the OpenEnv leaderboard.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Any

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("pip install openai python-dotenv")
    sys.exit(1)

# Load .env file
load_dotenv()

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.environment import SupportTriageEnv
from triage_env.models import Action, ActionType, TicketCategory, TicketPriority


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support agent. Your job is to triage incoming support tickets.

For each ticket you MUST respond with a JSON object only.

Schema:
{
  "action_type": "classify",
  "category": "billing" | "technical" | "account" | "shipping" | "returns" | "general" | "spam",
  "priority": "critical" | "high" | "medium" | "low" | "ignore",
  "team": "billing-team" | "tech-team" | "account-team" | "shipping-team" | "returns-team" | "support-team" | "security-team" | "legal-team" | "product-team" | "sales-team" | "partnerships-team" | "spam-filter",
  "reply_text": "string" | null,
  "reasoning": "string"
}

Priority guidelines:
- critical: SLA breach, outage, data loss, security breach, immediate physical need
- high: angry customer, major blocker, time-sensitive (within hours)
- medium: standard issue, needs attention within 1 business day
- low: minor question, feature request, can wait several days
- ignore: spam, test tickets, no action needed

Reply guidelines:
- Draft a reply ONLY if the ticket warrants a personal response
- Be empathetic for complaints/sensitive issues
- Be professional for enterprise/partnership inquiries
- Skip reply for spam/test tickets (set reply_text to null)
"""

def build_user_prompt(obs: dict, task_config: dict) -> str:
    ticket = obs["ticket"]
    hints  = obs.get("hints", [])
    prompt = f"""TICKET #{obs['step'] + 1} of {obs['inbox_total']}
Task: {obs['task_id']}
Inbox remaining: {obs['inbox_remaining']}
Episode reward so far: {obs['episode_reward_so_far']:.3f}

--- TICKET ---
ID:      {ticket['id']}
From:    {ticket['sender_email']}
Subject: {ticket['subject']}
Body:
{ticket['body']}
--------------
"""
    if hints:
        prompt += f"\nHints: {'; '.join(hints)}"
    prompt += "\n\nProvide the JSON action:"
    return prompt


def parse_llm_action(content: str) -> Optional[Action]:
    """Parse JSON from LLM response, tolerating markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (fences)
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    try:
        data = json.loads(content)
        # Handle enum validation manually or let Pydantic handle it
        return Action(
            action_type=ActionType(data.get("action_type", "classify")),
            category=TicketCategory(data["category"]) if data.get("category") else None,
            priority=TicketPriority(data["priority"])  if data.get("priority") else None,
            team=data.get("team"),
            reply_text=data.get("reply_text"),
            reasoning=data.get("reasoning"),
        )
    except Exception as e:
        print(f"    ⚠ Parse error: {e}\n    Raw: {content[:200]}")
        return None


def run_task(
    client: OpenAI,
    env: SupportTriageEnv,
    task_id: str,
    model_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one full episode on a task and return the summary."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}  |  Model: {model_name}")
    print(f"{'='*60}")

    obs = env.reset(task_id)
    obs_dict = obs.model_dump()
    done = False
    step = 0

    while not done:
        ticket = obs_dict["ticket"]
        if verbose:
            print(f"\n  Step {step+1}: [{ticket['id']}] {ticket['subject'][:60]}")

        task_config = {
            "requires_routing": task_id != "task1-easy",
            "requires_reply":   task_id == "task3-hard",
        }
        user_msg = build_user_prompt(obs_dict, task_config)

        # Call OpenAI (Hugging Face endpoint)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                response_format={ "type": "json_object" },
                max_tokens=800
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"    ⚠ API error: {e}")
            content = "{}"

        action = parse_llm_action(content)
        if action is None:
            # Fallback: skip with penalty
            action = Action(action_type=ActionType.SKIP)

        result = env.step(action)

        if verbose:
            r = result.reward
            print(f"    → cat={action.category} pri={action.priority} "
                  f"team={action.team} | reward={r.total:+.3f}")
            if action.reasoning:
                print(f"    reasoning:  {action.reasoning}")
            if action.reply_text:
                print(f"    reply_text: {action.reply_text}")

        obs_dict = result.observation.model_dump()
        done = result.done
        step += 1
        time.sleep(0.5)  

    summary = env.episode_summary()
    print(f"\n  ── Episode Summary ──")
    print(f"  Score:             {summary.score:.4f}")
    print(f"  Category acc:      {summary.category_accuracy:.4f}")
    print(f"  Priority acc:      {summary.priority_accuracy:.4f}")
    print(f"  Routing acc:       {summary.routing_accuracy:.4f}")
    print(f"  Total reward:      {summary.total_reward:.4f}")
    print(f"  Duration:          {summary.duration_seconds:.1f}s")
    return summary.model_dump()


def main():
    parser = argparse.ArgumentParser(description="OpenEnv baseline runner (OpenAI format)")
    parser.add_argument("--model",   default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--task",    default="all", choices=["all", "task1-easy", "task2-medium", "task3-hard"])
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("ERROR: Set HF_TOKEN environment variable (or put it in a .env file).")
        sys.exit(1)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1/",
        api_key=api_key
    )
    
    env = SupportTriageEnv()

    tasks = ["task1-easy", "task2-medium", "task3-hard"] if args.task == "all" else [args.task]

    results = {}
    for task_id in tasks:
        results[task_id] = run_task(client, env, task_id, args.model, args.verbose)

    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS — {args.model}")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Score':>8} {'Cat':>8} {'Pri':>8} {'Route':>8}")
    print(f"  {'-'*52}")
    for tid, r in results.items():
        print(f"  {tid:<20} {r['score']:>8.4f} {r['category_accuracy']:>8.4f} "
              f"{r['priority_accuracy']:>8.4f} {r['routing_accuracy']:>8.4f}")

    # Save results
    out_file = "baseline_results.json"
    with open(out_file, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
