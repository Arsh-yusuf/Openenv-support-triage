---
title: Customer Support Ticket Triage
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# OpenEnv: Customer Support Ticket Triage

[![openenv](https://img.shields.io/badge/openenv-v1.0.0-6b5df5)](https://github.com/openenv)
[![license](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-f59e0b)](https://huggingface.co/spaces/openenv/support-triage-v1)

A fully spec-compliant OpenEnv environment for training and evaluating AI agents on **real-world customer support ticket triage**. Agents must classify, prioritize, route, and respond to incoming support tickets — mimicking the work of a real support operations centre.

---

## Why This Environment

Customer support triage is a high-value, high-volume task at every company. It requires:
- **Multi-label classification** (category + priority simultaneously)
- **Contextual routing** (which team handles this?)
- **Natural language generation** (drafting empathetic, accurate replies)
- **Ambiguity resolution** (real tickets are messy)

This makes it an ideal benchmark for language model agents: it's complex enough to be meaningful, concrete enough to grade programmatically, and representative of real-world deployment conditions.

---

## Environment Description

The environment simulates an inbox of customer support tickets arriving for processing. At each step, the agent receives a ticket (subject, body, sender) and must take an action on it. The episode ends when all tickets in the inbox have been processed.

**Key design choices:**
- Shaped rewards provide signal throughout the episode (not just at the end)
- Priority scoring uses adjacency — a one-level mistake is penalised less than a two-level mistake
- Reply quality is graded heuristically without requiring an external LLM
- Spam detection carries a bonus to incentivise precision (low false positive rate)

---

## Action Space

```python
class Action(BaseModel):
    action_type: ActionType        # classify | route | reply | escalate | close | skip
    category:    TicketCategory    # billing | technical | account | shipping | returns | general | spam
    priority:    TicketPriority    # critical | high | medium | low | ignore
    team:        Optional[str]     # billing-team | tech-team | account-team | ...
    reply_text:  Optional[str]     # Free-text reply draft
    reasoning:   Optional[str]     # Optional chain-of-thought
```

**Available teams:** `billing-team`, `tech-team`, `account-team`, `shipping-team`, `returns-team`, `support-team`, `security-team`, `legal-team`, `product-team`, `sales-team`, `partnerships-team`, `spam-filter`

---

## Observation Space

```python
class Observation(BaseModel):
    step:                   int
    ticket:                 Ticket       # id, subject, body, sender_email, received_at
    inbox_remaining:        int
    inbox_total:            int
    actions_taken:          List[dict]   # last 3 actions (context window)
    episode_reward_so_far:  float
    task_id:                str
    hints:                  List[str]    # coaching hints (routing required? reply needed?)
```

---

## Reward Function

| Component | Value | Condition |
|---|---|---|
| Category correct | +0.40 | Exact match |
| Spam caught | +0.50 | Correct spam classification |
| Priority exact | +0.30 | Exact match |
| Priority adjacent | +0.15 | One level off (e.g., HIGH vs MEDIUM) |
| Routing correct | +0.20 | Correct team assignment |
| Reply quality | +0.00–0.50 | Heuristic score: length, empathy, no placeholders |
| Escalation appropriate | +0.05 | Escalating a CRITICAL/HIGH ticket |
| Speed bonus | +0.10 | Action submitted in < 5 seconds |
| Skip penalty | −0.30 | Skipping a ticket |
| Wrong escalation | −0.20 | Escalating a LOW/IGNORE ticket |

Rewards are clipped to `[-1.0, 1.0]` per step.

---

## Tasks

### Task 1 — Easy: Basic Triage
- **5 tickets** with clear, unambiguous signals
- Required: `category`, `priority`
- Metrics: category accuracy, priority accuracy
- Pass threshold: ≥ 0.80

```
t1-001: Billing dispute (clear dollar amount discrepancy)
t1-002: Spam (obvious phishing email)
t1-003: Account lockout (explicit "can't log in")
t1-004: Shipping delay (explicit tracking issue)
t1-005: Technical bug (app crash after update)
```

### Task 2 — Medium: Ambiguous Triage + Routing
- **8 tickets** with mixed signals; correct routing required
- Required: `category`, `priority`, `team`
- Metrics: category + priority + routing accuracy (partial credit for adjacent priority)
- Pass threshold: ≥ 0.70

Notable ambiguities:
- Churn-risk vs billing question (t2-001, t2-003)
- Follow-up on existing ticket with no category signal (t2-006)
- Developer API question vs support question (t2-007)

### Task 3 — Hard: Full Pipeline
- **12 tickets** covering edge cases, sensitive situations, and non-English input
- Required: `category`, `priority`, `team`, `reply_text`
- Metrics: category + priority + routing + reply quality
- Pass threshold: ≥ 0.65

Notable challenges:
- Sensitive contexts: bereavement refund, GDPR complaint, security breach
- Non-standard: partnership inquiry, Spanish-language request, internal test ticket
- Time-sensitive: enterprise outage 2 hours before board demo

---

## Baseline Scores (meta-llama/Llama-3.1-8B-Instruct, temperature=0.0)

| Task | Score | Category | Priority | Routing | Reply |
|---|---|---|---|---|---|
| task1-easy | **0.84** | 1.00 | 0.80 | — | — |
| task2-medium | **0.67** | 0.875 | 0.625 | 0.625 | — |
| task3-hard | **0.58** | 0.833 | 0.583 | 0.500 | 0.650 |

---

## Setup

### Local Python

```bash
git clone https://github.com/your-org/openenv-support-triage
cd openenv-support-triage
pip install -r requirements.txt

# Start the API server
python server.py
# → http://localhost:7860

# Run baseline (complies with hackathon format)
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py
```

### Docker

```bash
docker build -t openenv-support-triage .
docker run -p 7860:7860 openenv-support-triage

# With HF token and custom endpoint for evaluation
docker run -p 7860:7860 -e HF_TOKEN=hf_... -e API_BASE_URL=... -e MODEL_NAME=... openenv-support-triage
```

### Python SDK (no server)

```python
from triage_env.environment import SupportTriageEnv
from triage_env.models import Action, ActionType, TicketCategory, TicketPriority

env = SupportTriageEnv()
obs = env.reset("task1-easy")

while True:
    # Your agent logic here
    action = Action(
        action_type=ActionType.CLASSIFY,
        category=TicketCategory.BILLING,
        priority=TicketPriority.HIGH,
        reasoning="Invoice discrepancy mentioned",
    )
    result = env.step(action)
    print(f"Reward: {result.reward.total:+.3f}")
    if result.done:
        break

summary = env.episode_summary()
print(f"Final score: {summary.score:.4f}")
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit one action |
| GET | `/state` | Current environment state |
| GET | `/summary` | End-of-episode grader results |
| GET | `/info` | Environment metadata |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

---

## Project Structure

```
openenv-support-triage/
├── triage_env/
│   ├── models.py          # Typed Pydantic models
│   ├── environment.py     # Core SupportTriageEnv class
│   ├── tickets.py         # Synthetic ticket dataset
│   └── reward.py          # Shaped reward function
├── inference.py           # Standard hackathon entry point
├── graders/
│   └── graders.py         # Task1/2/3 graders (0.0–1.0)
├── baseline/
│   └── run_baseline.py    # Legacy baseline runner (reference only)
├── static/
│   └── index.html         # Demo UI
├── server.py              # FastAPI HTTP server
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile             # Container definition
├── requirements.txt
└── README.md
```

---

## OpenEnv Spec Compliance

- [x] Typed `Observation`, `Action`, `Reward` Pydantic models
- [x] `reset(task_id)` → initial observation
- [x] `step(action)` → (observation, reward, done, info)
- [x] `state()` → full mutable environment state
- [x] `openenv.yaml` with metadata, action/observation space, baseline scores
- [x] Minimum 3 tasks with agent graders (easy → medium → hard)
- [x] Shaped reward function with partial progress signals
- [x] Baseline inference script with reproducible scores
- [x] Dockerfile + Hugging Face Spaces deployment
- [x] `tagged: openenv`

---

## License

MIT — see LICENSE

## Citation

```bibtex
@misc{openenv-support-triage-2024,
  title  = {OpenEnv: Customer Support Ticket Triage},
  author = {OpenEnv Demo},
  year   = {2024},
  url    = {https://huggingface.co/spaces/openenv/support-triage-v1}
}
```
