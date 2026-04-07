"""
Synthetic ticket dataset.
Each ticket has ground-truth labels used only by the grader.
"""

from __future__ import annotations
from typing import List, Dict, Any
from triage_env.models import Ticket, TicketCategory, TicketPriority

# We store ground truth separately so it's never leaked in the Ticket object
GROUND_TRUTH: Dict[str, Dict[str, Any]] = {}

def _t(tid, subject, body, sender, cat, pri, team, received="2024-03-15T09:00:00Z", meta=None) -> Ticket:
    GROUND_TRUTH[tid] = {
        "category": cat,
        "priority": pri,
        "team": team,
    }
    return Ticket(
        id=tid,
        subject=subject,
        body=body,
        sender_email=sender,
        received_at=received,
        metadata=meta or {},
    )


# ─── TASK 1 — Easy: 5 tickets, clear signals ─────────────────────────────────

TASK1_TICKETS: List[Ticket] = [
    _t("t1-001",
       "Invoice #4821 shows wrong amount",
       "Hi, I was charged $149 but my plan is $99/month. Please fix this immediately. Order ref: INV-4821.",
       "alice@example.com",
       TicketCategory.BILLING, TicketPriority.HIGH, "billing-team"),

    _t("t1-002",
       "FREE IPHONE!! Click here",
       "Congratulations! You have been selected. Click the link to claim your prize: http://scam.xyz",
       "noreply@spam123.biz",
       TicketCategory.SPAM, TicketPriority.IGNORE, "spam-filter"),

    _t("t1-003",
       "Cannot log into my account",
       "I reset my password three times but still get 'invalid credentials'. I need access urgently for a presentation tomorrow.",
       "bob@corp.com",
       TicketCategory.ACCOUNT, TicketPriority.HIGH, "account-team"),

    _t("t1-004",
       "Where is my order #ORD-7732?",
       "I placed an order 2 weeks ago and tracking shows it has been stuck in transit for 10 days. Please advise.",
       "carol@home.net",
       TicketCategory.SHIPPING, TicketPriority.MEDIUM, "shipping-team"),

    _t("t1-005",
       "App crashes on startup",
       "After the latest update, your iOS app crashes immediately on launch. iPhone 15 Pro, iOS 17.3.",
       "dev@startup.io",
       TicketCategory.TECHNICAL, TicketPriority.HIGH, "tech-team"),
]


# ─── TASK 2 — Medium: 8 tickets, ambiguous signals ───────────────────────────

TASK2_TICKETS: List[Ticket] = [
    _t("t2-001",
       "Question about my subscription",
       "Hi there, I've been a customer for 3 years. I noticed a $5 increase on my last bill. Was there a price change? I'd like to understand before deciding whether to stay.",
       "loyal@customer.com",
       TicketCategory.BILLING, TicketPriority.MEDIUM, "billing-team"),

    _t("t2-002",
       "Integration stopped working",
       "Our Salesforce integration broke after your API v2 sunset. We're losing deal data. 200+ reps affected.",
       "cto@bigcorp.com",
       TicketCategory.TECHNICAL, TicketPriority.CRITICAL, "tech-team"),

    _t("t2-003",
       "How do I export my data?",
       "I'm thinking of switching to a competitor. Before I do, I want to export all my data. What's the process?",
       "maybe@leaving.com",
       TicketCategory.ACCOUNT, TicketPriority.MEDIUM, "account-team"),

    _t("t2-004",
       "Return request - birthday gift",
       "I received a duplicate birthday gift and would like to return one. It's been 35 days since purchase, but it's still in the box.",
       "giftreceiver@mail.com",
       TicketCategory.RETURNS, TicketPriority.LOW, "returns-team"),

    _t("t2-005",
       "URGENT: Security breach on my account",
       "I received a login notification from Lagos, Nigeria at 3am. I did NOT log in. I think my account has been hacked. Please lock it NOW.",
       "scared@user.com",
       TicketCategory.ACCOUNT, TicketPriority.CRITICAL, "security-team"),

    _t("t2-006",
       "Re: Re: Re: Follow up on ticket #3301",
       "Still waiting for an update on this. It's been 2 weeks. Is anyone there?",
       "waiting@forever.com",
       TicketCategory.GENERAL, TicketPriority.MEDIUM, "support-team"),

    _t("t2-007",
       "API rate limit documentation unclear",
       "Your docs say 1000 req/min but I'm getting 429s at 800. Is the limit per-endpoint or global?",
       "api@developer.io",
       TicketCategory.TECHNICAL, TicketPriority.MEDIUM, "tech-team"),

    _t("t2-008",
       "Discount not applied at checkout",
       "I used code SAVE20 but only got 10% off. Order #ORD-9981. Please refund the difference.",
       "deal@hunter.com",
       TicketCategory.BILLING, TicketPriority.MEDIUM, "billing-team"),
]


# ─── TASK 3 — Hard: 12 tickets, subtle signals + reply drafting required ─────

TASK3_TICKETS: List[Ticket] = [
    _t("t3-001",
       "Partnership inquiry",
       "Hi, I'm the VP of Partnerships at TechCorp (ARR $50M). We'd love to explore a strategic partnership. Who should I connect with?",
       "vp@techcorp.com",
       TicketCategory.GENERAL, TicketPriority.HIGH, "partnerships-team",
       meta={"requires_reply": True, "reply_tone": "professional_warm"}),

    _t("t3-002",
       "Charge I don't recognise",
       "There's a $0.01 charge from your company on my card. I've never signed up for anything. Is this a scam?",
       "worried@user.net",
       TicketCategory.BILLING, TicketPriority.MEDIUM, "billing-team",
       meta={"requires_reply": True, "reply_tone": "reassuring"}),

    _t("t3-003",
       "My entire team can't access the dashboard",
       "Since 8am this morning, all 47 users in our enterprise account get a 503 error. We have a board demo in 2 hours.",
       "enterprise@client.com",
       TicketCategory.TECHNICAL, TicketPriority.CRITICAL, "tech-team",
       meta={"requires_reply": True, "reply_tone": "urgent_empathetic"}),

    _t("t3-004",
       "Accessibility issue with your website",
       "I use a screen reader and the checkout flow is broken — the 'Place Order' button has no ARIA label. This is a legal compliance issue.",
       "accessibility@advocate.org",
       TicketCategory.TECHNICAL, TicketPriority.HIGH, "tech-team",
       meta={"requires_reply": True, "reply_tone": "empathetic_action_oriented"}),

    _t("t3-005",
       "Refund for deceased account holder",
       "My mother passed away last month. She had an annual subscription with 8 months remaining. Can we get a pro-rated refund?",
       "family@bereaved.com",
       TicketCategory.BILLING, TicketPriority.HIGH, "billing-team",
       meta={"requires_reply": True, "reply_tone": "compassionate"}),

    _t("t3-006",
       "Potential data privacy violation - GDPR",
       "I submitted a data deletion request 45 days ago (ref: GDPR-2291). Under GDPR you had 30 days to comply. No action taken. I am filing a formal complaint.",
       "legal@privacy.eu",
       TicketCategory.ACCOUNT, TicketPriority.CRITICAL, "legal-team",
       meta={"requires_reply": True, "reply_tone": "formal_apologetic"}),

    _t("t3-007",
       "Feature suggestion",
       "Would be great to have dark mode. Just a thought!",
       "casual@user.com",
       TicketCategory.GENERAL, TicketPriority.LOW, "product-team",
       meta={"requires_reply": False}),

    _t("t3-008",
       "Wrong item shipped - urgent medical equipment",
       "I ordered a blood pressure monitor (Model BP-200) for my elderly father. You shipped a kitchen thermometer. He needs the BP monitor today.",
       "caregiver@family.net",
       TicketCategory.SHIPPING, TicketPriority.CRITICAL, "shipping-team",
       meta={"requires_reply": True, "reply_tone": "urgent_empathetic"}),

    _t("t3-009",
       "Re: Auto-renewal notice",
       "DO NOT RENEW MY SUBSCRIPTION. I've sent 3 emails and called twice. If you charge me again I will dispute with my bank and report to the FTC.",
       "angry@customer.com",
       TicketCategory.BILLING, TicketPriority.HIGH, "billing-team",
       meta={"requires_reply": True, "reply_tone": "de_escalating"}),

    _t("t3-010",
       "Curious about enterprise pricing",
       "We're a 500-person company currently on 10 seats. Wondering if there are volume discounts if we expand.",
       "procurement@enterprise.com",
       TicketCategory.BILLING, TicketPriority.MEDIUM, "sales-team",
       meta={"requires_reply": True, "reply_tone": "sales_supportive"}),

    _t("t3-011",
       "App in Spanish?",
       "Hola, ¿tienen la aplicación en español? Soy de México y preferiría usarla en mi idioma.",
       "usuario@mexico.mx",
       TicketCategory.GENERAL, TicketPriority.LOW, "product-team",
       meta={"requires_reply": True, "reply_tone": "helpful", "language": "es"}),

    _t("t3-012",
       "Test ticket please ignore",
       "Testing 1 2 3",
       "qa@internal.company.com",
       TicketCategory.SPAM, TicketPriority.IGNORE, "spam-filter",
       meta={"requires_reply": False}),
]


TASKS: Dict[str, Dict[str, Any]] = {
    "task1-easy": {
        "name": "Basic Triage",
        "description": "Classify and prioritize 5 clearly labelled support tickets.",
        "difficulty": "easy",
        "tickets": TASK1_TICKETS,
        "requires_routing": False,
        "requires_reply": False,
        "max_reward_per_ticket": 2.0,
    },
    "task2-medium": {
        "name": "Ambiguous Triage + Routing",
        "description": "Classify, prioritize, and route 8 ambiguous tickets to the correct team.",
        "difficulty": "medium",
        "tickets": TASK2_TICKETS,
        "requires_routing": True,
        "requires_reply": False,
        "max_reward_per_ticket": 3.0,
    },
    "task3-hard": {
        "name": "Full Triage + Routing + Reply",
        "description": "Handle 12 complex tickets: classify, prioritize, route, and draft appropriate replies.",
        "difficulty": "hard",
        "tickets": TASK3_TICKETS,
        "requires_routing": True,
        "requires_reply": True,
        "max_reward_per_ticket": 5.0,
    },
}
