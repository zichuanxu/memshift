# Registration supplement to v4: a strong reader and definitions in the user's words (2026-10-16)

Status: registered before any request of these two blocks. It adds two blocks to
`prereg/v4_round3.md` and changes nothing in it or in any earlier registration. Shared settings
(probes, checkpoints, intervals, the anchor rule, no multiplicity correction) are v4's.

What was known when this was written, and shaped it:
- Block Y's predictions come from the exploratory store/reader decomposition of the committed main
  runs (`runs/stage6_ex_round3/analysis/store_reader.json`, decision log 2026-10-16). On the 42 items
  other than the definitions, after S_def, B×D at D=1 splits into a stored-value part of +20.5 pp
  [+11.7, +29.1] and a reader part of −21.1 pp [−31.7, −10.8]; after S_del, +7.2 pp and −8.6 pp;
  after S_src, +2.8 pp and −3.4 pp. On definition items the reader part is 0.0 pp after S_src and
  S_def. No answer from any reader other than Qwen3.5-9B on these cells was seen.
- Block V's sets pair with block G-b's, which were queued but had no result when this was written.
- The interim refresh ratios of block U (U1 holds on 20 worlds) were known. Neither block below bears
  on U.

| Block | Question (reviewer) | Worlds | Model calls |
|---|---|---|---|
| Y | Is the interaction on the other 42 items hidden by the 9B reader? (R2 W5, R1 W2, W7) | 1000-1019 | reader only: Claude Sonnet 5.5 |
| V | Do the writer's programs keep the interaction when the definition is stated in the user's own words? (R1 W4, Q4; R2 Q2) | 1000-1019 | program requests (12 definition versions × 9 sets) |

## Y. A strong reader on the same stores

The reader is Claude Sonnet 5.5 (`claude-sonnet-5-5`, Anthropic Messages API), with the reader
request of every cell unchanged: the same system prompt, evidence (4,096-token budget), question,
instructions, 256 output tokens and JSON schema (sent as structured output). Settings:
`thinking: between_tools`, so it does not think before answering, as Qwen3.5-9B reads with thinking
off; the API default sampling (the 5.5 models take no temperature), so answers are not
deterministic; every response is cached and released. Everything the reader sees comes from the
committed main runs: writes and programs are cache hits of the main writer, so every cell's store,
and so every reader context, is the committed one.

Cells `D0B0`, `D0B1_source`, `D1B0`, `D1B1_source`, `D1sB0`, `D1sB1_source`, worlds 1000-1019,
branches src, def, orphan, discriminative probes (as E1). Anchor: every probe's context hash equals the
committed run's, and every writer and program request is a cache hit; otherwise the run is void.
Validity: at most 2% of Sonnet's answers are unreadable or cut off at the output limit; otherwise Y1-Y3
are not judged.

- **Y1** (the 42 other items, D=1, after S_def): B×D with Sonnet as reader is positive with a 95%
  interval above 0.
- **Y2** (the 42 other items, D=1, after S_del): the same.
- **Y3** (definition items, D=1, after S_src and after S_def): B×D with Sonnet as reader is within
  5 pp of B×D with Qwen3.5-9B as reader (point estimate of the paired difference over worlds), in
  both.
- Reported without a threshold: the same at D_m; the other items after S_src; the store/reader
  decomposition with Sonnet as reader (`ex_round3` store/reader analysis, reader = Sonnet); the number
  of correct stored "no value" items that the reader misreads.
- Readings: if Y1 holds, the interaction is not confined to the six definitions; the 9B reader
  hides it on the other items, and the paper says so where it reports the class split. If Y1 fails,
  the reader's losses are not specific to a small reader, and the class split stands as reported.
  Y3 tests that the definition-item result does not depend on the reader.

## V. Definitions in the user's own words

The D_m program request (`holdmem/program.py`) states the definition in effect in a fixed, codified
form ("An important contact is a person who attended at least 5 of the user's one-off calendar
meetings (not recurring series) in the last 90 days."). In block V the request replaces that sentence
by the sentence below, written by the authors before any V request: the same information (sources,
thresholds, windows, conditions), phrased as the user would say it. Nothing else in the request
changes: the record fields, the declared sources and the task text are the memory's own data model,
which a deployed memory has, and are kept. The probes, the reader requests and the gold are
unchanged.

| Definition | v1 | v2 |
|---|---|---|
| important_contact | "Count someone as an important contact if they've been at five or more of my one-off meetings in the past 90 days. Recurring series don't count." | "Count someone as an important contact if they've been at two or more of my one-off meetings in the past 30 days. Recurring series don't count." |
| close_friend | "Anyone I've had six or more chat messages with in the past 30 days, theirs and mine together, is a close friend." | "Anyone I've had four or more chat messages with in the past 14 days, theirs and mine together, is a close friend." |
| active_project | "A project is active as long as it still has a task I haven't finished." | "A project is active if any of its tasks, finished or not, was created or changed in the past 14 days." |
| working_hours | "My working hours are whatever my latest 'Working hours' note says." | "Go by my one-off work meetings in the past 30 days: from the earliest start to the latest end, widened to whole hours." |
| vip_sender | "Treat a contact as a VIP sender if they've emailed me four or more times in the past 30 days." | "Treat a contact as a VIP sender if they've emailed me three or more times in the past 30 days, or at least once if I've starred them." |
| overdue_task | "A task is overdue if I haven't finished it and its due date is before today." | "Only high-priority tasks count as overdue: not finished, and due before today." |

Nine program sets, as in G-b and paired with G-b's sets of the same writer and sample index:
Qwen3.5-9B (temperature 0.7, samples 0, 1, 2), Claude Sonnet 5.5 and Claude Opus 5.5 (G-b's settings
for Claude programs, samples 0, 1, 2). Each set is replayed in `D1sB0` and `D1sB1_source` on
1000-1019, branches src, def, orphan; only the definition items' programs change, every other request
is a cache hit. Anchor: as G-b (D1s cells at k=0 for the other items).

- **V1**: B×D at D_m on definition items after S_def, over the nine sets, is positive with the two-way
  interval (worlds × program sets) above 0.
- **V2**: the same after S_src.
- Reported without a threshold: per writer, the paired difference V minus G-b (same writer and
  samples) in Update on changed definition items at `D1sB1_source` and in the programs' agreement
  with gold before the shift; the interaction after S_del.
- Readings: if V1 and V2 hold, design rules 2 and 4 apply to a definition the user states in their own
  words, given the memory's data model. A statement of intent without its thresholds and windows has no
  single gold value under this design and stays untested; the paper says so.

## Budget

Y: about 27,000 reader requests of about 4,300 input tokens each, about US$330 at Sonnet 5.5's list
prices; no GPU. V: about 110 program requests (under US$10) and about 3-5 GPU hours of reader requests
on definition items. Neither block is cut before blocks U, H, G-a and N.
