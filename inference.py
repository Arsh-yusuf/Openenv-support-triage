#!/usr/bin/env python3
"""
OpenEnv RL Challenge: inference.py
Satisfies the hackathon submission format requirements.

Usage:
    export HF_TOKEN=hf_...
    python inference.py
"""

import os
import sys
import json
import time
import traceback
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Add current directory to path so we can import from env/graders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file
load_dotenv()

from triage_env.environment import SupportTriageEnv
from triage_env.models import Action, ActionType, TicketCategory, TicketPriority

# ─── Environment Variables (Hackathon Core Requirements) ─────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

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
"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
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
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    try:
        data = json.loads(content)
        return Action(
            action_type=ActionType(data.get("action_type", "classify")),
            category=TicketCategory(data.get("category")) if data.get("category") else None,
            priority=TicketPriority(data.get("priority"))  if data.get("priority") else None,
            team=data.get("team"),
            reply_text=data.get("reply_text"),
            reasoning=data.get("reasoning"),
        )
    except Exception:
        return None

# ─── Main Inference Loop ──────────────────────────────────────────────────────

def run_task(env: SupportTriageEnv, task_id: str):
    """Run one task and emit standardized stdout formatting."""
    rewards = []
    steps = 0
    success = False
    
    # ── [START] ──
    print(f"[START] task={task_id} env=support-triage model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id)
        obs_dict = obs.model_dump()
        done = False
        
        while not done:
            user_msg = build_user_prompt(obs_dict)
            
            error_msg = "null"
            action_type_str = "null"
            reward_val = 0.0
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={ "type": "json_object" },
                    max_tokens=600
                )
                content = response.choices[0].message.content
                action = parse_llm_action(content)
                if not action:
                    action = Action(action_type=ActionType.SKIP)
                    error_msg = "parse_error"
                
                action_type_str = action.action_type.value
                result = env.step(action)
                
                obs_dict = result.observation.model_dump()
                done = result.done
                reward_val = result.reward.total
                
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                reward_val = -0.1  # System penalty for failure
                done = True

            steps += 1
            rewards.append(reward_val)
            
            # ── [STEP] ──
            done_str = "true" if done else "false"
            print(f"[STEP] step={steps} action={action_type_str} reward={reward_val:.2f} done={done_str} error={error_msg}", flush=True)
            
            if done:
                break
        
        # Summary check
        summary = env.episode_summary()
        success = summary.score >= 0.70  # Arbitrary success threshold for log
        
    except Exception as e:
        # Emit one error step if reset fails
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={str(e)}", flush=True)
    
    finally:
        # ── [END] ──
        success_str = "true" if success else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

def main():
    env = SupportTriageEnv()
    tasks = ["task1-easy", "task2-medium", "task3-hard"]
    
    for tid in tasks:
        run_task(env, tid)
        print("", flush=True) # Blank line between episodes

if __name__ == "__main__":
    main()
