# Module 2 Learnings: AI Agents & Sovereign Systems

---

## Assignment 1: Sovereign Agent — Week 1

### Exercise 1: Context Engineering

- **"Lost in the Middle" effect**: LLM answer retrieval drops when relevant information sits in the middle of long context. Beginning and end positions perform better.
- **Prompt formatting matters**: Instruction order, delimiters, and output constraints materially change model accuracy.
- **XML-style structure**: Using clear separators and structured prompts improves reliability across vendors (OpenAI, Anthropic).
- **Context placement strategies**: Sandwiching critical info at the beginning or end of prompts yields more reliable extractions than burying it mid-context.
- **Evaluation mindset**: One-off prompt checks should evolve into repeatable eval workflows with dataset-driven regression tracking.

### Exercise 2: LangGraph Research Agent (The Headless Automator)

- **ReAct pattern**: Interleaving reasoning (`Thought:`) with actions (`Action:`) allows the agent to decide its own tool-calling sequence.
- **LangGraph StateGraph**: Graph-based execution with conditional edges and explicit state handling.
- **Tool implementation**: `venue_search`, `get_edinburgh_weather`, `calculate_catering_cost`, and `generate_event_flyer` — each returning structured `ToolResult` objects.
- **Autonomous planning**: The agent receives a high-level task ("find a pub for 160 people") and decides which tools to call, in what order, without explicit scripting.
- **Failure handling**: When the first-choice venue is unavailable, the agent must find an alternative without human guidance.
- **Graph visualization**: The agent's execution path can be rendered as a Mermaid diagram for debugging.

### Exercise 3: Rasa Pro CALM (The Digital Employee)

- **Deterministic flow contract**: `flows.yml` guarantees slot collection order (`guest_count → vegan_count → deposit_amount_gbp → action_validate_booking`). The LLM cannot reorder or skip steps.
- **Python enforces business rules; LLM handles language**: The LLM extracts slot values; Python custom actions (`ActionValidateBooking`) apply hard constraints.
- **Time-based cutoff guards**: Business rules like "no confirmations after 16:45" are enforced in Python, not inferred by the LLM.
- **Deposit and party-size validation**: `MAX_DEPOSIT_GBP` and max party size caps trigger escalation with clear reasons.
- **Out-of-scope deflection**: `handle_out_of_scope` displays `utter_out_of_scope`, then offers to resume the paused flow with slot state preserved.
- **Two-terminal architecture**: `rasa run actions` (port 5055) handles custom logic; `rasa run --enable-api` (port 5005) handles conversation.
- **Why CALM for confirmations**: Auditable, deterministic, every decision traceable — essential when "every word could cost money."

### Exercise 4: MCP Shared Tool Layer

- **MCP contract**: The server (`mcp_venue_server.py`) is the single source of truth. Changing `"status": "available"` → `"full"` immediately changes all clients' results without code edits.
- **Schema is not optional**: Without `args_schema`, `StructuredTool` silently degrades to text generation. The LLM writes out what it would call, but tools never execute.
- **Async bridge pattern**: Mixing sync LangGraph with async MCP requires a `ThreadPoolExecutor` boundary — each `asyncio.run()` call runs in a fresh thread.
- **System prompts for function calling**: Without a system prompt instructing one-tool-at-a-time calling, Llama-3.3-70B batches all intended calls into a single JSON text block.
- **Trace extraction**: LangChain's `AIMessage.tool_calls` attribute (not just Anthropic-style content blocks) must be checked to capture tool invocations.
- **Tool discovery over code changes**: New tools registered in the MCP server are automatically discoverable by all clients.

---

## Assignment 2: Pub Booking — Ex5 through Ex9

### Ex5: Edinburgh Research Scenario (The Loop Half)

- **Four tools**: `venue_search`, `get_weather`, `calculate_cost`, `generate_flyer` — each logs arguments and outputs to `_TOOL_CALL_LOG`.
- **Parallel-safe annotations**: Read-only tools marked `parallel_safe=True`; `generate_flyer` (file write) marked `False`.
- **Dataflow integrity check (`verify_dataflow`)**: Every fact in the final flyer (venue name, price, weather condition) must trace back to a tool call. Hallucinated facts fail the check.
- **Fabrication test**: Deliberately editing the flyer (e.g., changing £540 to £9999) causes `verify_dataflow` to report the unverified fact.
- **Deterministic vs. real mode**: `FakeLLMClient` with scripted trajectory for fast testing; `OpenAICompatibleClient` with `--real` for actual LLM behavior.
- **Tool-spiral detection**: Qwen-3-32B may make 5+ `venue_search` calls with increasingly desperate params. The diagnostic histogram reveals uncalled tools.
- **Session directory structure**: Every run creates `sess_<id>/` with `SESSION.md`, `session.json`, `workspace/`, `logs/trace.jsonl`, `extras/tickets/`, and `ipc/`.

### Ex6: Rasa Structured Half

- **`StructuredHalf` subclass**: Routes booking intent dicts into Rasa via HTTP POST and maps responses back to `HalfResult`.
- **Rasa flows**: `confirm_booking` (happy path), `resume_from_loop` (mid-scenario handoff), `request_research` (structured rejection → back to loop).
- **`ActionValidateBooking`**: Checks deposit <= £300 and party size <= 8; returns rejection reason to the flow if either fails.
- **Validator (`validator.py`)**: Normalizes loose booking data into Rasa's REST message shape — parses £ into int, canonicalizes dates, handles timezone and venue_id.
- **Three-terminal setup**: Terminal 1 (`make rasa-actions`), Terminal 2 (`make rasa-serve`), Terminal 3 (`make ex6-real`).
- **Mock mode**: `make ex6` uses stdlib mock server for development without a Rasa license.

### Ex7: Handoff Bridge (Bidirectional Round-Trip)

- **Bridge orchestration**: `HandoffBridge.run()` manages loop → structured → loop → structured → completion, max 3 rounds.
- **Atomic file IPC**: Handoff messages written to `ipc/handoff_to_*.json`. At most one handoff file visible at any time (fail-closed rule).
- **Rejection flow**: Loop finds venue → structured rejects (party > 8) → bridge builds reverse task → loop re-researches → second structured attempt succeeds.
- **Session state machine**: Clear `session.state_changed` events for each transition (`loop → structured`, `structured → loop`, `structured → complete`).
- **Grader's planted failure**: The structured half may always reject; the bridge must catch and report this rather than looping forever.

### Ex8: Voice Pipeline

- **Manager persona**: Llama-3.3-70B-Instruct with a gruff Edinburgh pub manager system prompt.
- **STT → Agent → TTS round-trip**:
  - Speechmatics real-time STT over websocket
  - Agent processes text and generates response
  - Rime Arcana/ElevenLabs TTS → MP3 → pydub decode → sounddevice playback
- **Text mode (primary gradeable)**: `--text` reads from stdin, prints responses. No API keys needed.
- **Voice mode (bonus)**: `--voice` requires `SPEECHMATICS_KEY` and `RIME_API_KEY`, plus microphone access.
- **Graceful degradation**: Missing `SPEECHMATICS_KEY` falls back to text mode with a visible warning instead of crashing.
- **Trace events**: Every utterance logged as `voice.utterance_in` and `voice.utterance_out` with correct event types.

### Ex9: Reflection

- **Grounded answers**: Every answer cites specific `sess_xxxx` IDs, ticket IDs, and trace lines from actual runs.
- **Planner handoff analysis**: Understanding what signal caused the planner to assign a subgoal to the structured half.
- **Dataflow integrity in practice**: Describing specific scenarios where `verify_dataflow` catches failures a human reviewer wouldn't.
- **Production failure prediction**: Naming exactly one sovereign-agent primitive (ticket state machine, manifest discipline, IPC atomic rename, SessionQueue retry) and one failure mode it would surface.

---

## Cross-Cutting Themes

1. **Two-agent architecture**: The same problem (pub booking) requires two genuinely different architectures — a headless automator for open-ended research and a digital employee for deterministic confirmation.
2. **Deterministic vs. generative**: CALM flows guarantee behavior; LangGraph agents explore. Neither is universally better — the skill is knowing which to reach for.
3. **Dataflow integrity**: Every fact in an agent's output must trace back to a verifiable source. Without this, LLM hallucinations go undetected.
4. **Schema-first tool design**: Tools without schemas silently degrade to text generation. Always define `args_schema` when wrapping external capabilities.
5. **Session-as-directory**: Every run produces a complete artifact directory (`sess_<id>/`) with human-readable summaries and machine-parseable traces.
6. **Graceful degradation**: Production agents must handle missing APIs, rejected bookings, and unavailable services without crashing.
7. **Atomic IPC**: Handoff between agent halves uses atomic file writes to prevent race conditions and ensure fail-closed behavior.
8. **MCP as shared infrastructure**: A single tool server serves multiple clients (LangGraph agent, Rasa action, voice pipeline), eliminating code duplication and ensuring consistency.
