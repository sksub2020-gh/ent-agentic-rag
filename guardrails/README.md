Here's the complete picture:
1. New /guardrails/ module — 4 files:
2. base.py — GuardBase interface + GuardAction enum (PASS / BLOCK / REDACT / WARN). Implement GuardBase to add any custom guard.
3. input_guards.py — 4 guards: PromptInjectionGuard (regex-based jailbreak detection), TopicRelevanceGuard (configurable blocked topics), InputLengthGuard, PiiDetectionGuard. All configurable, warn-only by default.
4. output_guards.py — 4 guards: PiiRedactionGuard (actively rewrites output), ToxicityGuard, HallucinationFlagGuard (catches uncertainty phrases), AnswerCompletenessGuard.
5. guard_runner.py — runs stacks of guards in order, first BLOCK wins, REDACT actions accumulate. Also contains the two LangGraph nodes (input_guard_node, output_guard_node).
- Updated files: agents/state.py (added blocked, block_reason, guard_warnings), agents/graph.py (new graph flow with guardrail nodes at both ends, input guard short-circuits to END if blocked).

### New graph flow:
```
input_guard → (blocked? → END) → router → rag → critique → (retry?) → output_guard → END
```

### Tests need no services:
```
python -m pytest tests/test_guardrails.py -v
```

### To customize for your domain — just pass your own guard stacks to build_rag_graph():
```
app = build_rag_graph(
    input_guards=[InputLengthGuard(), PromptInjectionGuard(), MyCustomGuard()],
    output_guards=[PiiRedactionGuard(), AnswerCompletenessGuard()],
)
```

That completes all 6 phases. Your full project structure is now:
```
config/ core/ ingestion/ retrieval/ agents/ evaluation/ guardrails/ tests/
```
