# Dynamic tools benchmark

### Model Comparison
┌───────────────────┬─────────────┬───────────────┬──────────────────┐
│      Metric       │ gpt-4o-mini │ grok-4.1-fast │ claude-haiku-4.5 │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Pass rate         │ 6/10 (60%)  │ 7/10 (70%)    │ 9/10 (90%)       │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Fails             │ 4           │ 0             │ 0                │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Partials          │ 0           │ 3             │ 1                │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Total calls       │ 44          │ 74            │ 43               │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Avg overhead      │ 77%         │ 78%           │ 63%              │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Redundant browses │ varied      │ 5-7/case      │ 0-3/case         │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Tokens            │ 49k         │ 117k          │ 95k              │
├───────────────────┼─────────────┼───────────────┼──────────────────┤
│ Total time        │ 89s         │ 111s          │ 90s              │
└───────────────────┴─────────────┴───────────────┴──────────────────┘
Haiku's strengths:
- 9/10 pass — only missed calendar_booking because it noticed a time conflict and flagged it instead of double-booking
(arguably the right call)
- 63% meta overhead — lowest of the three, meaning more calls go to actual business tools
- 43 total calls — same as mini but actually completes the work
- Zero redundant browses on simple cases — browses once, loads, acts
- Perfect protocol compliance on every single case
The efficiency gap is real: Haiku does in 3 calls (browse→load→use) what Grok does in 8-14 and what mini often can't
finish at all.