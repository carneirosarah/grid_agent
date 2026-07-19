"""Step 8 — LangGraph workflow: separation of planning from execution.

The graph wires the previous steps into an agent loop:

                    +-----------+
        START ----> |   plan    |  (LLM: WireReply via structured output)
                    +-----------+
                          |
                    +-----------+
                    | validate  |  (wire_to_plan + validate_plan, no LLM)
                    +-----------+
                     |    |    |
          clarify <--+    |    +--> invalid: back to `plan` with the error
          (END)           |         report, at most MAX_REPAIR_ATTEMPTS,
                          |         then give up (END, outcome "error")
                          v
                    +-----------+
                    |  preview  |  (engine runs on a copy; nothing committed)
                    +-----------+
                          |
                         END  -> user decides accept/reject via the API

Two separations enforced here:

1. **Planning vs execution.** The LLM only ever produces a `WireReply`
   inside `plan`. Applying operations happens in `preview` via the
   deterministic engine — and even that is only a staged preview; the
   commit happens later, outside the graph, when the human accepts.

2. **Repair vs conversation.** When validation fails, the error report is
   appended to a *transient* copy of the history for the retry, and never
   saved into the session's chat history — the user shouldn't scroll past
   the model's internal mistakes. Every attempt is traced, though.

Human-in-the-loop note: accept/reject is deliberately *outside* the graph
(plain session methods called by the API) instead of a LangGraph interrupt.
With a stateless HTTP frontend this is simpler and equally safe; the trade
is discussed in DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import MAX_REPAIR_ATTEMPTS
from .llm import Planner, PlannerError, build_table_context
from .metrics import PassRateCounter
from .schemas import Plan, WireReply, wire_to_plan
from .state import PendingChange, TableSession
from .trace import Tracer
from .validator import validate_plan


class AgentState(TypedDict, total=False):
    """State flowing through the graph for ONE user instruction."""
    history: list[dict[str, str]]   # transient conversation incl. repairs
    wire_reply: WireReply | None    # raw structured LLM output
    plan: Plan | None               # resolved plan (post-validation)
    errors: list[str]               # last validation failure report
    attempts: int                   # planning attempts so far
    outcome: Literal["preview", "clarify", "error"] | None
    message: str                    # user-facing reply text


@dataclass
class TurnResult:
    """What one full graph run produced, for the API/frontend."""
    outcome: Literal["preview", "clarify", "error"]
    message: str
    pending: PendingChange | None = None


def build_graph(session: TableSession, planner: Planner, tracer: Tracer,
                semantic: PassRateCounter | None = None):
    """Compile the plan->validate->preview graph.

    Nodes close over (session, planner, tracer): the graph state itself
    stays small, while the heavyweight objects (the DataFrame, the LLM
    client) live outside it.

    `semantic` tallies the Semantic Validation Pass Rate (see metrics.py
    for the counting rules); the API passes one server-wide counter, and
    callers that don't care (tests, scripts) get a throwaway.
    """
    semantic = semantic or PassRateCounter()

    # -- node: plan ---------------------------------------------------------
    def plan_node(state: AgentState) -> AgentState:
        """Ask the LLM for a structured reply. On a retry, the validator's
        error report is appended as an extra user turn (transient)."""
        history = list(state["history"])
        if state.get("errors"):
            history.append({
                "role": "user",
                "text": ("Your previous plan failed validation. Fix these "
                         "errors and resend the complete corrected plan:\n- "
                         + "\n- ".join(state["errors"])),
            })
        context = build_table_context(session.df)   # always the live table
        try:
            reply, call = planner.plan(context, history)
        except PlannerError as exc:
            tracer.log("planner_error", error=str(exc))
            return {"wire_reply": None, "outcome": "error",
                    "message": f"The planning model is unavailable: {exc}"}
        # Observability first (model, prompt refs, tokens, cost, latency),
        # then the content the call produced.
        tracer.log("model_call",
                   attempt=state.get("attempts", 0), **asdict(call))
        tracer.log("llm_reply",
                   attempt=state.get("attempts", 0),
                   reply=reply.model_dump())
        return {"wire_reply": reply, "attempts": state.get("attempts", 0) + 1}

    # -- node: validate -----------------------------------------------------
    def validate_node(state: AgentState) -> AgentState:
        """Structural + semantic validation. Never calls the LLM."""
        reply = state["wire_reply"]

        if reply.intent == "clarify":
            question = reply.clarifying_question or \
                "Could you rephrase that instruction?"
            tracer.log("clarification_asked", question=question)
            return {"outcome": "clarify", "message": question}

        structural_plan, errors = wire_to_plan(reply)
        if not errors:
            # A structural plan reached the semantic validator — its
            # verdict feeds the Semantic Validation Pass Rate. Structural
            # failures above never get here and are deliberately not
            # counted (wall 2, not wall 3 — see metrics.py).
            result = validate_plan(session.df, structural_plan)
            semantic.record(result.ok)
            if result.ok:
                tracer.log("plan_validated", plan=result.plan.model_dump())
                return {"plan": result.plan, "errors": []}
            errors = result.errors
        tracer.log("validation_failed",
                   attempt=state["attempts"], errors=errors)
        return {"errors": errors, "plan": None}

    # -- node: preview ------------------------------------------------------
    def preview_node(state: AgentState) -> AgentState:
        """Stage the validated plan as a pending preview (no commit)."""
        pending = session.propose(state["plan"])
        stats = "; ".join(
            f"{r.description}: {r.cells_changed} cell(s) changed"
            for r in pending.op_results)
        message = (f"{pending.summary}\n{stats}. "
                   "Review the highlighted preview and accept or reject.")
        return {"outcome": "preview", "message": message}

    # -- node: give_up ------------------------------------------------------
    def give_up_node(state: AgentState) -> AgentState:
        """Repair budget exhausted: fail loudly rather than guess."""
        tracer.log("repair_exhausted", errors=state.get("errors", []))
        details = "\n- ".join(state.get("errors", ["unknown error"]))
        return {"outcome": "error",
                "message": ("I couldn't produce a valid plan for that "
                            f"instruction. Last problems:\n- {details}\n"
                            "Try rephrasing the request.")}

    # -- routing ------------------------------------------------------------
    def after_plan(state: AgentState) -> str:
        return END if state.get("outcome") == "error" else "validate"

    def after_validate(state: AgentState) -> str:
        if state.get("outcome") == "clarify":
            return END
        if state.get("plan") is not None:
            return "preview"
        # Invalid plan: retry until the budget is spent.
        if state["attempts"] <= MAX_REPAIR_ATTEMPTS:
            return "plan"
        return "give_up"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("preview", preview_node)
    graph.add_node("give_up", give_up_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, ["validate", END])
    graph.add_conditional_edges("validate", after_validate,
                                ["preview", "plan", "give_up", END])
    graph.add_edge("preview", END)
    graph.add_edge("give_up", END)
    return graph.compile()


def run_turn(session: TableSession, planner: Planner, tracer: Tracer,
             user_message: str,
             semantic: PassRateCounter | None = None) -> TurnResult:
    """Process one chat message end to end.

    Owns conversation-history hygiene: the user turn is recorded before the
    run; the assistant's user-facing reply is recorded after. Repair-loop
    noise stays out of the durable history (see module docstring).
    """
    tracer.log("user_message", text=user_message)
    session.history.append({"role": "user", "text": user_message})

    app = build_graph(session, planner, tracer, semantic)
    final: AgentState = app.invoke({
        "history": list(session.history),
        "attempts": 0,
        "errors": [],
    })

    outcome: str = final.get("outcome") or "error"
    message: str = final.get("message") or "Something went wrong."
    session.history.append({"role": "model", "text": message})
    tracer.log("turn_finished", outcome=outcome, message=message)

    return TurnResult(
        outcome=outcome,  # type: ignore[arg-type]
        message=message,
        pending=session.pending if outcome == "preview" else None,
    )
