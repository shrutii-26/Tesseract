from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, Optional
import httpx
import json
import re
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from database import (
    get_commitments,
    save_commitment,
    update_commitment_status,
    delete_commitment,
    get_learning_items,
    save_learning_item,
    update_learning_status,
    delete_learning_item,
    add_guardian_history,
    get_guardian_history,
    get_guardian_patterns,
    get_interests,
    save_interest,
    delete_interest,
    get_rules,
    save_rule,
    delete_rule,
)

app = FastAPI(title="Tesseract", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tesseract-wine.vercel.app", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

MODEL = "llama-3.1-8b-instant"

pending_approvals: dict = {}

# ── Shared HTTP client (created once at startup, reused for all requests) ────
http_client: httpx.AsyncClient = None


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=60)


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


# ═══════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════


class TriageRequest(BaseModel):
    messages: list[dict]


class DraftRequest(BaseModel):
    message: dict
    tone: str = "professional"


class ResearchRequest(BaseModel):
    question: str


class ExtractCommitmentsRequest(BaseModel):
    text: str


class GuardianRequest(BaseModel):
    notifications: list[dict]
    context: dict


class SocialCurateRequest(BaseModel):
    posts: list[dict]
    interests: list[str]


class LearningOrchestrateRequest(BaseModel):
    items: list[dict]
    energy: str
    available_minutes: int


class BriefRequest(BaseModel):
    messages: list = []
    notifications: list = []
    posts: list = []
    interests: list = []
    energy: str = "medium"
    available_minutes: int = 20
    thread_id: str = "default"


class HITLResponse(BaseModel):
    thread_id: str
    decisions: list


# DB models
class CommitmentIn(BaseModel):
    id: str
    task: str
    owner: str = "You"
    urgency: str = "medium"
    context: str = ""
    status: str = "pending"
    days_stale: int = 0


class LearningItemIn(BaseModel):
    id: str
    title: str
    url: str = ""
    type: str = "article"
    topic: str = ""
    minutes: int = 15
    status: str = "not-started"
    days_stale: int = 0


class StatusUpdate(BaseModel):
    status: str


class InterestIn(BaseModel):
    interest: str


class RuleIn(BaseModel):
    rule: str


class GuardianHistoryIn(BaseModel):
    sender: str
    channel: str
    decision: str


# ═══════════════════════════════════════════════════════════════════
# LLM HELPER
# ═══════════════════════════════════════════════════════════════════


async def llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    # Uses the shared http_client instead of creating a new one per call
    response = await http_client.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"Groq error: {data}")
    return data["choices"][0]["message"]["content"]


async def with_timeout(coro, label: str, seconds: int = 20):
    """Wrap a coroutine with a timeout so one slow agent can't hang the brief."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return {"error": f"{label} timed out after {seconds}s"}


def parse_json(raw: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

TRIAGE_PROMPT = """You are an inbox triage assistant. Identify TOP 3 messages needing attention today.
Respond ONLY with JSON, no extra text:
{{
  "priority": [{{"id": "<id>", "reason": "<one sentence>", "urgency": "high", "suggested_action": "<action>"}}],
  "rest": [{{"id": "<id>", "classification": "auto-archive", "reason": "<reason>"}}]
}}
Messages:
{messages}
JSON:"""

DRAFT_PROMPT = """Write a short reply. Be {tone}. Under 80 words. No subject line, just the body.
From: {sender}
Subject: {subject}
Message: {body}
Reply:"""

RESEARCH_PROMPT = """You are a research assistant. Answer with a clear verdict based on search results.
Question: {question}
Search Results:
{results}
Respond ONLY with JSON, no extra text:
{{
  "verdict": "<one clear sentence>",
  "reasoning": "<2-3 sentences>",
  "confidence": <0-100>,
  "pros": ["<pro 1>", "<pro 2>", "<pro 3>"],
  "cons": ["<con 1>", "<con 2>", "<con 3>"],
  "sources": ["<url 1>", "<url 2>", "<url 3>"]
}}
JSON:"""

EXTRACT_COMMITMENTS_PROMPT = """Extract every commitment or promise from this text.
Look for: "I'll", "I will", "remind me", "I need to", "I should", "I'll follow up", "will do", etc.
Respond ONLY with JSON, no extra text:
{{
  "commitments": [{{"task": "<task>", "owner": "<owner>", "urgency": "high|medium|low", "context": "<context>"}}]
}}
If none found: {{"commitments": []}}
Text:
{text}
JSON:"""

GUARDIAN_PROMPT = """You are an Attention Guardian. Evaluate each notification: ALLOW or BLOCK.
Rules: {rules}
Past decisions: {history}
Guidelines:
- BLOCK: newsletters, promotions, social media, non-urgent pings, automated alerts
- ALLOW: boss/manager, urgent deadlines, someone blocked on the user, emergencies
Notifications:
{notifications}
Respond ONLY with JSON, no extra text:
{{"decisions": [{{"id": "<id>", "decision": "allow", "reason": "<reason>", "confidence": 90}}]}}
JSON:"""

SOCIAL_CURATOR_PROMPT = """You are a social media curator. Score posts and pick top 3 to engage with.
User interests: {interests}
HIGH signal (7-10): original insight, personal news, directly relevant to interests
MEDIUM signal (4-6): mildly interesting, general industry news
LOW signal (1-3): promotional, engagement bait, generic content
Posts:
{posts}
Respond ONLY with JSON, no extra text:
{{
  "scored": [{{"id": "<id>", "signal_score": 8, "signal_label": "high", "reason": "<reason>"}}],
  "top_picks": [{{"id": "<id>", "why_engage": "<why>", "suggested_action": "reply", "draft": "<draft>"}}]
}}
JSON:"""

LEARNING_PROMPT = """You are a learning coach. Build a {available}-minute sprint for {energy} energy.
Items:
{items}
Respond ONLY with JSON, no extra text:
{{
  "sprint": [{{"id": "<id>", "order": 1, "nudge": "<nudge>", "minutes": 15}}],
  "total_minutes": 15,
  "stale_guilt": [{{"id": "<id>", "message": "<guilt>"}}],
  "skipped": [{{"id": "<id>", "reason": "<reason>"}}]
}}
JSON:"""


# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE ROUTES
# ═══════════════════════════════════════════════════════════════════


# Commitments
@app.get("/db/commitments")
async def db_get_commitments():
    return get_commitments()


@app.post("/db/commitments")
async def db_save_commitment(item: CommitmentIn):
    save_commitment(item.dict())
    return {"ok": True}


@app.patch("/db/commitments/{id}")
async def db_update_commitment(id: str, body: StatusUpdate):
    update_commitment_status(id, body.status)
    return {"ok": True}


@app.delete("/db/commitments/{id}")
async def db_delete_commitment(id: str):
    delete_commitment(id)
    return {"ok": True}


# Learning items
@app.get("/db/learning")
async def db_get_learning():
    return get_learning_items()


@app.post("/db/learning")
async def db_save_learning(item: LearningItemIn):
    save_learning_item(item.dict())
    return {"ok": True}


@app.patch("/db/learning/{id}")
async def db_update_learning(id: str, body: StatusUpdate):
    update_learning_status(id, body.status)
    return {"ok": True}


@app.delete("/db/learning/{id}")
async def db_delete_learning(id: str):
    delete_learning_item(id)
    return {"ok": True}


# Guardian history
@app.get("/db/guardian/history")
async def db_get_guardian_history():
    return get_guardian_history()


@app.post("/db/guardian/history")
async def db_add_guardian_history(body: GuardianHistoryIn):
    add_guardian_history(body.sender, body.channel, body.decision)
    return {"ok": True}


@app.get("/db/guardian/patterns")
async def db_get_guardian_patterns():
    return get_guardian_patterns()


# Interests
@app.get("/db/interests")
async def db_get_interests():
    return get_interests()


@app.post("/db/interests")
async def db_save_interest(body: InterestIn):
    save_interest(body.interest)
    return {"ok": True}


@app.delete("/db/interests/{interest}")
async def db_delete_interest(interest: str):
    delete_interest(interest)
    return {"ok": True}


# Rules
@app.get("/db/rules")
async def db_get_rules():
    return get_rules()


@app.post("/db/rules")
async def db_save_rule(body: RuleIn):
    save_rule(body.rule)
    return {"ok": True}


@app.delete("/db/rules/{rule}")
async def db_delete_rule(rule: str):
    delete_rule(rule)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════
# AGENT ROUTES
# ═══════════════════════════════════════════════════════════════════


@app.post("/triage")
async def triage(req: TriageRequest):
    prompt = TRIAGE_PROMPT.format(messages=json.dumps(req.messages, indent=2))
    raw = await llm(prompt, max_tokens=400)
    return parse_json(raw) or {"error": "Could not parse response", "raw": raw}


@app.post("/draft")
async def draft(req: DraftRequest):
    msg = req.message
    prompt = DRAFT_PROMPT.format(
        tone=req.tone,
        sender=msg.get("from", "Unknown"),
        subject=msg.get("subject", "No subject"),
        body=msg.get("body", ""),
    )
    raw = await llm(prompt, max_tokens=200, temperature=0.7)
    return {"draft": raw.strip()}


@app.post("/research")
async def research(req: ResearchRequest):
    # FIX: switched from "advanced" to "basic" — advanced depth adds 8-15s latency
    # and isn't needed since we only use the top snippet content anyway
    tavily_response = await http_client.post(
        TAVILY_URL,
        json={
            "api_key": TAVILY_API_KEY,
            "query": req.question,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": False,
        },
        timeout=20,
    )
    results = tavily_response.json().get("results", [])
    if not results:
        return {"error": "No search results found"}
    formatted = ""
    for i, r in enumerate(results[:5], 1):
        formatted += f"\n[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', '')[:300]}\n"
    prompt = RESEARCH_PROMPT.format(question=req.question, results=formatted)
    raw = await llm(prompt, max_tokens=500)
    result = parse_json(raw)
    if result and not result.get("sources"):
        result["sources"] = [r.get("url", "") for r in results[:3]]
    return result or {"error": "Could not parse response", "raw": raw}


@app.post("/extract-commitments")
async def extract_commitments(req: ExtractCommitmentsRequest):
    prompt = EXTRACT_COMMITMENTS_PROMPT.format(text=req.text)
    raw = await llm(prompt, max_tokens=400)
    return parse_json(raw) or {"error": "Could not parse response", "raw": raw}


@app.post("/guardian")
async def guardian(req: GuardianRequest):
    rules_text = "\n".join(req.context.get("rules", [])) or "No custom rules."
    history = req.context.get("history", [])
    history_text = (
        "\n".join(
            [
                f"- {h['from']} ({h.get('channel','unknown')}): {h['decision']}"
                for h in history[-20:]
            ]
        )
        or "No history."
    )
    prompt = GUARDIAN_PROMPT.format(
        rules=rules_text,
        history=history_text,
        notifications=json.dumps(req.notifications, indent=2),
    )
    raw = await llm(prompt, max_tokens=300)
    result = parse_json(raw) or {"decisions": []}
    for d in result.get("decisions", []):
        notif = next((n for n in req.notifications if n["id"] == d["id"]), {})
        add_guardian_history(
            notif.get("from", d["id"]), notif.get("channel", "unknown"), d["decision"]
        )
    return result


@app.post("/social-curate")
async def social_curate(req: SocialCurateRequest):
    interests_text = (
        ", ".join(req.interests) if req.interests else "technology, business"
    )
    prompt = SOCIAL_CURATOR_PROMPT.format(
        interests=interests_text, posts=json.dumps(req.posts, indent=2)
    )
    raw = await llm(prompt, max_tokens=500, temperature=0.2)
    return parse_json(raw) or {"error": "Could not parse response", "raw": raw}


@app.post("/learning-orchestrate")
async def learning_orchestrate(req: LearningOrchestrateRequest):
    prompt = LEARNING_PROMPT.format(
        energy=req.energy,
        available=req.available_minutes,
        items=json.dumps(req.items, indent=2),
    )
    raw = await llm(prompt, max_tokens=400, temperature=0.2)
    return parse_json(raw) or {"error": "Could not parse response", "raw": raw}


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH DAILY BRIEF
# ═══════════════════════════════════════════════════════════════════


class BriefState(TypedDict):
    messages: list
    notifications: list
    commitments: list
    posts: list
    learning_items: list
    interests: list
    energy: str
    available_minutes: int
    rules: list
    run_triage: bool
    run_guardian: bool
    run_social: bool
    run_learning: bool
    triage_result: dict
    guardian_result: dict
    commitments_result: dict
    social_result: dict
    learning_result: dict
    hitl_required: bool
    hitl_items: list
    hitl_approved: list
    hitl_rejected: list
    brief: dict
    error: str


def router_node(state: BriefState) -> BriefState:
    return {
        **state,
        "run_triage": len(state.get("messages", [])) > 0,
        "run_guardian": len(state.get("notifications", [])) > 0,
        "run_social": len(state.get("posts", [])) > 0,
        "run_learning": len(state.get("learning_items", [])) > 0,
        "hitl_required": False,
        "hitl_items": [],
        "hitl_approved": [],
        "hitl_rejected": [],
    }


async def _triage_agent(state):
    prompt = TRIAGE_PROMPT.format(messages=json.dumps(state["messages"], indent=2))
    raw = await llm(prompt, max_tokens=400)
    return parse_json(raw) or {"priority": [], "rest": []}


async def _guardian_agent(state):
    rules_text = "\n".join(state.get("rules", [])) or "No custom rules."
    prompt = GUARDIAN_PROMPT.format(
        rules=rules_text,
        history="No history.",
        notifications=json.dumps(state["notifications"], indent=2),
    )
    raw = await llm(prompt, max_tokens=300)
    return parse_json(raw) or {"decisions": []}


async def _commitments_agent(state):
    commitments = state.get("commitments", [])
    pending = [c for c in commitments if c.get("status") == "pending"]
    overdue = [
        c
        for c in commitments
        if c.get("days_stale", 0) >= 3 and c.get("status") == "pending"
    ]
    return {"pending": pending, "overdue": overdue, "total": len(pending)}


async def _social_agent(state):
    interests_text = ", ".join(state.get("interests", ["technology"]))
    prompt = SOCIAL_CURATOR_PROMPT.format(
        interests=interests_text, posts=json.dumps(state["posts"], indent=2)
    )
    raw = await llm(prompt, max_tokens=500, temperature=0.2)
    return parse_json(raw) or {"top_picks": [], "scored": []}


async def _learning_agent(state):
    prompt = LEARNING_PROMPT.format(
        available=state.get("available_minutes", 20),
        energy=state.get("energy", "medium"),
        items=json.dumps(state["learning_items"], indent=2),
    )
    raw = await llm(prompt, max_tokens=400, temperature=0.2)
    return parse_json(raw) or {"sprint": [], "stale_guilt": [], "total_minutes": 0}


async def parallel_agents_node(state: BriefState) -> BriefState:
    tasks = {}
    if state.get("run_triage"):
        tasks["triage"] = _triage_agent(state)
    if state.get("run_guardian"):
        tasks["guardian"] = _guardian_agent(state)
    if state.get("commitments"):
        tasks["commitments"] = _commitments_agent(state)
    if state.get("run_social"):
        tasks["social"] = _social_agent(state)
    if state.get("run_learning"):
        tasks["learning"] = _learning_agent(state)

    if tasks:
        keys = list(tasks.keys())
        # FIX: each agent wrapped with a 20s timeout so one slow LLM call
        # can't hold up the entire brief
        results = await asyncio.gather(
            *[with_timeout(tasks[k], k, seconds=20) for k in keys],
            return_exceptions=True,
        )
        result_map = {
            k: (r if not isinstance(r, Exception) else {"error": str(r)})
            for k, r in zip(keys, results)
        }
    else:
        result_map = {}

    return {
        **state,
        "triage_result": result_map.get("triage", {"priority": [], "rest": []}),
        "guardian_result": result_map.get("guardian", {"decisions": []}),
        "commitments_result": result_map.get(
            "commitments", {"pending": [], "overdue": []}
        ),
        "social_result": result_map.get("social", {"top_picks": [], "scored": []}),
        "learning_result": result_map.get(
            "learning", {"sprint": [], "stale_guilt": [], "total_minutes": 0}
        ),
    }


def hitl_gate_node(state: BriefState) -> BriefState:
    hitl_items = []
    decisions = state.get("guardian_result", {}).get("decisions", [])
    for d in decisions:
        if d.get("confidence", 100) < 70:
            hitl_items.append(
                {
                    "type": "guardian_uncertain",
                    "id": d["id"],
                    "description": f"Guardian uncertain about '{d.get('id')}' (confidence: {d.get('confidence')}%)",
                    "current_decision": d["decision"],
                    "options": ["allow", "block"],
                }
            )
    overdue = state.get("commitments_result", {}).get("overdue", [])
    for c in overdue:
        if c.get("days_stale", 0) >= 7:
            hitl_items.append(
                {
                    "type": "overdue_commitment",
                    "id": c["id"],
                    "description": f"Commitment '{c.get('task')}' is {c.get('days_stale')} days stale",
                    "options": ["acknowledge", "drop"],
                }
            )
    return {**state, "hitl_required": len(hitl_items) > 0, "hitl_items": hitl_items}


async def synthesize_node(state: BriefState) -> BriefState:
    triage = state.get("triage_result", {})
    guardian = state.get("guardian_result", {})
    commitments = state.get("commitments_result", {})
    social = state.get("social_result", {})
    learning = state.get("learning_result", {})

    priority_count = len(triage.get("priority", []))
    allowed_count = len(
        [d for d in guardian.get("decisions", []) if d.get("decision") == "allow"]
    )
    blocked_count = len(
        [d for d in guardian.get("decisions", []) if d.get("decision") == "block"]
    )
    overdue_count = len(commitments.get("overdue", []))
    sprint_count = len(learning.get("sprint", []))
    social_picks = len(social.get("top_picks", []))

    summary_prompt = """Write a short punchy morning brief (3-4 sentences). Direct, human, not corporate. No bullet points.
Facts:
- {priority} emails need attention
- {allowed} notifications allowed, {blocked} blocked
- {overdue} commitments overdue
- Learning sprint: {sprint} items, {minutes} minutes
- {social} social posts worth engaging
Write it now:""".format(
        priority=priority_count,
        allowed=allowed_count,
        blocked=blocked_count,
        overdue=overdue_count,
        sprint=sprint_count,
        minutes=learning.get("total_minutes", 0),
        social=social_picks,
    )

    summary_raw = await llm(summary_prompt, max_tokens=200, temperature=0.7)

    brief = {
        "summary": summary_raw.strip(),
        "stats": {
            "priority_messages": priority_count,
            "notifications_allowed": allowed_count,
            "notifications_blocked": blocked_count,
            "overdue_commitments": overdue_count,
            "learning_minutes": learning.get("total_minutes", 0),
            "social_picks": social_picks,
        },
        "triage": triage,
        "guardian": guardian,
        "commitments": commitments,
        "social": social,
        "learning": learning,
        "hitl_pending": state.get("hitl_items", []),
        "agents_run": {
            "triage": state.get("run_triage", False),
            "guardian": state.get("run_guardian", False),
            "social": state.get("run_social", False),
            "learning": state.get("run_learning", False),
        },
    }
    return {**state, "brief": brief}


def route_after_hitl(state: BriefState) -> str:
    return "synthesize"


def build_graph():
    memory = MemorySaver()
    graph = StateGraph(BriefState)
    graph.add_node("router", router_node)
    graph.add_node("parallel_agents", parallel_agents_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "parallel_agents")
    graph.add_edge("parallel_agents", "hitl_gate")
    graph.add_conditional_edges(
        "hitl_gate", route_after_hitl, {"synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=memory)


brief_graph = build_graph()


@app.post("/daily-brief")
async def daily_brief(req: BriefRequest):
    commitments = get_commitments()
    learning_items = get_learning_items()
    interests = get_interests() or req.interests or ["technology", "business"]
    rules = get_rules()

    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = BriefState(
        messages=req.messages,
        notifications=req.notifications,
        commitments=commitments,
        posts=req.posts,
        learning_items=learning_items,
        interests=interests,
        energy=req.energy,
        available_minutes=req.available_minutes,
        rules=rules,
        run_triage=False,
        run_guardian=False,
        run_social=False,
        run_learning=False,
        triage_result={},
        guardian_result={},
        commitments_result={},
        social_result={},
        learning_result={},
        hitl_required=False,
        hitl_items=[],
        hitl_approved=[],
        hitl_rejected=[],
        brief={},
        error="",
    )
    try:
        result = await brief_graph.ainvoke(initial_state, config=config)
        brief = result["brief"]
        if brief.get("hitl_pending"):
            pending_approvals[req.thread_id] = {
                "items": brief["hitl_pending"],
                "state": result,
            }
            brief["hitl_thread_id"] = req.thread_id
        return brief
    except Exception as e:
        return {"error": str(e)}


@app.get("/hitl-pending/{thread_id}")
async def get_hitl_pending(thread_id: str):
    if thread_id not in pending_approvals:
        return {"items": []}
    return {"items": pending_approvals[thread_id]["items"]}


@app.post("/hitl-resolve")
async def hitl_resolve(req: HITLResponse):
    if req.thread_id not in pending_approvals:
        return {"error": "No pending approvals for this thread"}
    stored = pending_approvals[req.thread_id]
    state = stored["state"]
    guardian_decisions = state.get("guardian_result", {}).get("decisions", [])
    for human_decision in req.decisions:
        for d in guardian_decisions:
            if d["id"] == human_decision["id"]:
                d["decision"] = human_decision["decision"]
                d["human_override"] = True
    del pending_approvals[req.thread_id]
    updated_state = {
        **state,
        "guardian_result": {
            **state.get("guardian_result", {}),
            "decisions": guardian_decisions,
        },
    }
    final = await synthesize_node(updated_state)
    return final["brief"]


# ═══════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════


# FIX: removed live Groq API call from health check — it was burning rate
# limits and adding latency on every UptimeRobot ping (every 5 minutes)
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"name": "Tesseract", "version": "1.0.0", "status": "running", "agents": 6}
