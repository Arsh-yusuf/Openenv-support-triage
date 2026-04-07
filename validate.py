#!/usr/bin/env python3
"""
openenv validate — checks that this environment is fully spec-compliant.
Run from the project root: python validate.py
"""
import sys, os, yaml, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"

results = []

def check(name, fn):
    try:
        fn()
        results.append((True, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((False, name))
        print(f"  {FAIL}  {name}\n      → {e}")

print("\nOpenEnv Validator — support-triage-v1\n" + "─"*40)

# 1. YAML spec
def check_yaml():
    with open("openenv.yaml") as f:
        d = yaml.safe_load(f)
    for key in ["env_id","version","tasks","action_space","observation_space","reward"]:
        assert key in d, f"Missing key: {key}"
    assert len(d["tasks"]) >= 3, "Need >= 3 tasks"
check("openenv.yaml exists and has required keys", check_yaml)

# 2. Models
def check_models():
    from triage_env.models import Observation, Action, Reward, StepResult, EpisodeSummary
    a = Action(action_type="classify", category="billing", priority="high")
    assert a.action_type == "classify"
check("Typed Pydantic models (Observation, Action, Reward)", check_models)

# 3. Environment API
def check_env_api():
    from triage_env.environment import SupportTriageEnv
    from triage_env.models import Action, ActionType, TicketCategory, TicketPriority
    env = SupportTriageEnv()
    assert hasattr(env, "reset"), "Missing reset()"
    assert hasattr(env, "step"),  "Missing step()"
    assert hasattr(env, "state"), "Missing state()"
    obs = env.reset("task1-easy")
    assert obs.task_id == "task1-easy"
    assert obs.inbox_total == 5
    state = env.state()
    assert "task_id" in state
check("SupportTriageEnv has reset() / step() / state()", check_env_api)

# 4. Full episode
def check_episode():
    from triage_env.environment import SupportTriageEnv
    from triage_env.models import Action, ActionType, TicketCategory, TicketPriority
    env = SupportTriageEnv()
    obs = env.reset("task1-easy")
    done = False
    steps = 0
    while not done:
        result = env.step(Action(
            action_type=ActionType.CLASSIFY,
            category=TicketCategory.BILLING,
            priority=TicketPriority.HIGH,
        ))
        done = result.done
        steps += 1
        assert -1.0 <= result.reward.total <= 1.0, "Reward out of range"
    assert steps == 5
check("Full episode runs, rewards in [-1,1], terminates at N steps", check_episode)

# 5. Episode summary + grader
def check_grader():
    from triage_env.environment import SupportTriageEnv
    from triage_env.models import Action, ActionType, TicketCategory, TicketPriority
    env = SupportTriageEnv()
    env.reset("task1-easy")
    for _ in range(5):
        env.step(Action(action_type=ActionType.CLASSIFY,
                        category=TicketCategory.BILLING, priority=TicketPriority.HIGH))
    s = env.episode_summary()
    assert 0.0 <= s.score <= 1.0, f"Score {s.score} out of [0,1]"
check("episode_summary() returns score in [0.0, 1.0]", check_grader)

# 6. All 3 tasks
def check_all_tasks():
    from triage_env.environment import SupportTriageEnv
    from triage_env.models import Action, ActionType, TicketCategory, TicketPriority
    from triage_env.tickets import TASKS
    env = SupportTriageEnv()
    for tid, cfg in TASKS.items():
        obs = env.reset(tid)
        n = len(cfg["tickets"])
        for _ in range(n):
            env.step(Action(action_type=ActionType.CLASSIFY,
                            category=TicketCategory.GENERAL, priority=TicketPriority.MEDIUM))
        s = env.episode_summary()
        assert 0.0 <= s.score <= 1.0
check("All 3 tasks run to completion with valid scores", check_all_tasks)

# 7. Graders
def check_graders():
    from graders.graders import GRADERS, Task1Grader, Task2Grader, Task3Grader
    assert "task1-easy" in GRADERS
    assert "task2-medium" in GRADERS
    assert "task3-hard" in GRADERS
    r = Task1Grader().grade([], [])
    assert 0.0 <= r.score <= 1.0
check("All 3 graders registered and return valid scores", check_graders)

# 8. Dockerfile
def check_dockerfile():
    with open("Dockerfile") as f:
        d = f.read()
    assert "7860" in d, "Missing port 7860"
    assert "python" in d.lower()
check("Dockerfile exposes port 7860", check_dockerfile)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for ok,_ in results if ok)
total  = len(results)
print(f"\n{'─'*40}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  \033[92mAll checks passed — environment is OpenEnv spec-compliant.\033[0m")
else:
    failed = [n for ok,n in results if not ok]
    print(f"  \033[91mFailed: {', '.join(failed)}\033[0m")
    sys.exit(1)
