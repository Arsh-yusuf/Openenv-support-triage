---
title: OpenEnv Support Triage v1
emoji: 📬
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - nlp
  - customer-support
  - classification
  - reinforcement-learning
  - agent-environment
short_description: OpenEnv-compliant AI training environment for customer support triage
---

# OpenEnv: Customer Support Ticket Triage

See README.md for full documentation.

## Quick Start

```bash
# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1-easy"}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"classify","category":"billing","priority":"high","team":"billing-team"}'

# Get episode summary
curl http://localhost:7860/summary
```

Visit `/docs` for the interactive Swagger API explorer.
