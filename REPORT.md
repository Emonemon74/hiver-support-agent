# Delta Twitter Support Agent — Report

All numbers are from `reports/results.json` and `reports/judge_agreement.json`,
computed on a **100-example stratified subsample** of the 199-example golden set
(the free Groq tier caps you at 200k tokens/day/model; running the full agent +
judge needs ~1.2M). The subsample keeps every intent and the 55/45 auto/escalate
split; `data/eval_subset.json` lists the ids.

## 1. Problem framing

**Brand:** @Delta. **Task:** for an incoming customer tweet — (1) classify intent,
(2) draft a public reply grounded in how Delta historically answered similar
tweets, (3) decide auto-send vs. escalate, with a reason.

**What "good" means here.** Delta's support Twitter is a *triage and holding*
channel — the substantive fixes happen in DMs with account access. So a good
agent, in priority order:
1. never auto-sends something that misleads a customer or commits Delta to a
   policy / refund / compensation it wouldn't honour;
2. clears the genuinely safe traffic without a human — compliments, general
   policy questions, feedback acknowledgement (that is most of the volume);
3. escalates everything else *with a specific, actionable reason*.

The metric that matters is **false-auto rate** (should-escalate messages that were
auto-sent), then how much human load is removed.

**What I deliberately did not build:** no fine-tuning (few-shot + retrieval only);
no multi-turn dialogue state (opening message only); no live integrations (flight
status, PNR lookup, posting); no non-English (5% of Delta threads, filtered); no
generation of the DM conversation that follows an escalation.

## 2. System

```
tweet ─▶ classify (gpt-oss-120b / -20b, 8 intents + other, 5-shot)
      ─▶ retrieve (FAISS · bge-small-en-v1.5 local, top-5 past resolved threads)
      ─▶ draft   (grounded ONLY in retrieved replies; abstains → grounded=false)
      ─▶ route   (deterministic rules: safety / money / disruption / churn /
                  live-data / intent-risk / confidence  ▸ then an LLM
                  upgrade-only safety check that can turn auto→escalate)
```
Corpus = 6,000 older threads; golden set = 199 newer threads, hand-labelled,
time-split so a golden example's own resolution is never retrievable. Model stack
is free-tier: Groq `gpt-oss` (draft/classify), Groq `qwen3.8-27b` (judge — a
different lineage), local sentence-transformers (embeddings). Details:
`INTENTS.md`, `docs/golden_set_note.md`, `DECISIONS.md`.

## 3. Results vs. baselines

### 3a. Intent classification

| system | accuracy | macro-F1 |
|--------|---------:|---------:|
| trivial — majority class (`compliment`) | 0.24 | 0.04 |
| simple — kNN, leave-one-out over golden embeddings | 0.47 | 0.30 |
| **agent — gpt-oss few-shot** | **0.83** | **0.79** |

`compliment` is perfect (F1 1.00). Weakest: `booking_reservation` 0.67,
`checkin_boarding` 0.73, `loyalty_miles` 0.73. Dominant error: specific issues
phrased with frustration get absorbed into `complaint` (2 flight_disruption, 2
baggage, 1 booking, 1 seat, 1 loyalty → complaint), and `checkin_boarding` is
confused with `seat_upgrade` (3 of 7).

### 3b. Routing (auto vs. escalate) — the weak point

| system | acc | escalate-recall | **false-auto** | over-escalation |
|--------|----:|----------------:|---------------:|----------------:|
| trivial — always escalate | 0.45 | 1.00 | **0.00** | 1.00 |
| trivial — always auto | 0.55 | 0.00 | 1.00 | 0.00 |
| simple — escalate if intent-risk = high (predicted intent) | **0.65** | 0.53 | 0.47 | 0.26 |
| **agent router** | 0.59 | 0.62 | **0.38** | 0.44 |

**The engineered router loses to a one-line rule on accuracy** (0.59 vs 0.65) and
still auto-sends 38% of messages that should reach a human. It buys slightly
higher recall (0.62 vs 0.53) by over-escalating more (0.44 vs 0.26). Cohen's κ is
0.18 — barely above chance. This is the headline negative result; see failure
mode 1 and §6.

### 3c. Reply quality (LLM judge, 1–5)

| reply | overall | ground | help | tone | safety | halluc-flag |
|-------|--------:|-------:|-----:|-----:|-------:|------------:|
| trivial — canned holding reply | 2.77 | 1.98 | 1.68 | 2.70 | 4.75 | 0.10 |
| simple — nearest past reply, verbatim | 3.73 | 4.03 | 2.80 | 3.52 | 4.58 | 0.15 |
| **agent draft (all 100)** | **4.13** | 3.95 | 3.82 | 4.49 | 4.25 | 0.18 |
| agent draft — auto-sent only (n=48) | 4.31 | 4.25 | 3.98 | 4.60 | 4.40 | 0.17 |

The agent beats both baselines overall and is much stronger on *helpfulness* and
*tone* than the verbatim-nearest-neighbour reply — but it is **less safe** (4.25
vs 4.58) and gets more hallucination flags, because it generates rather than
copies. Auto-sent replies score higher across the board (the router does keep the
worst drafts back), but 8 of 48 auto-sent replies are still hallucination-flagged.
Agent draft vs. Delta's actual reply, mean cosine: 0.62.

## 4. Is the judge trustworthy?

39 agent replies scored by a human (the author) and by the judge (`qwen3.8-27b`)
on the same rubric.

| dimension | quad-weighted κ | Spearman ρ | mean abs error | human / judge mean |
|-----------|---------------:|-----------:|---------------:|-------------------:|
| groundedness | 0.47 | 0.50 | 0.90 | 3.79 / 3.77 |
| helpfulness | 0.45 | 0.28 | 0.87 | 3.64 / 3.79 |
| tone | 0.52 | 0.35 | 0.49 | 4.44 / 4.46 |
| safety | 0.43 | 0.62 | 0.80 | 4.28 / 4.00 |
| **pooled** | **0.49** | **0.46** | **0.76** | — |

**Read:** the judge is *well-calibrated in aggregate* (dimension means within 0.3
of the human) but agrees only *moderately* on individual replies (pooled κ 0.49,
Landis–Koch "moderate"). Helpfulness is where it tracks the human worst
(ρ 0.28). It is slightly stricter on safety than the human. Conclusion: trust the
**aggregate** comparisons in §3c, not any single per-reply score.

## 5. Top 5 failure modes

1. **Router under-escalates specific complaints (false-auto 38%).** When a
   customer describes a concrete account-level problem in a frustrated tone, the
   classifier calls it `complaint` (medium risk) and the router auto-sends. Real
   examples auto'd that should have escalated: *"gate staff gave my seat away and
   then I got attitude"* (E-ACCOUNT), *"why is Delta charging for a lap infant AND
   checked bags"* (E-MONEY — the keyword regex is `charged?`, so "charging" never
   matched), *"2 hours for a callback to change my flight tomorrow"*
   (E-DISRUPTION). *Hypothesis:* medium-risk intents should default to escalate
   unless the message is clearly generic venting; keyword rules are too brittle
   for money/disruption detection.

2. **Draft invents specifics — phone numbers, policy, compensation (≈18%
   flagged, ≈8–10% hard fabrications).** e.g. a fabricated *"888-750-3284"* support
   line; *"Gold members don't qualify for complimentary upgrades — only Platinum
   Elite"* (factually wrong for Delta); *"Enjoy those extra SkyMiles!"* (unbacked
   compensation); a copied fake agent signature *"\*ABN <URL>"* pulled from a
   precedent reply. *Hypothesis:* the grounding instruction is not enforced;
   precedent that itself contains signatures/links leaks into the draft. Needs a
   post-generation check that every named entity/number appears in the retrieved
   context.

3. **Classifier collapses distinct issues into `complaint`, and confuses
   `checkin_boarding` with `seat_upgrade`.** 7 of the ~24 misclassifications route
   into `complaint`; 3 of 7 `checkin_boarding` examples become `seat_upgrade`
   (both involve seats and the app). *Hypothesis:* the taxonomy boundary between
   "I have a service grievance" and "I have a specific problem with X" is under-
   specified in the prompt; add contrastive few-shot pairs.

4. **Agent reads non-problems as problems and over-apologises.** *"This year's
   diamond benefits are outstanding 💎 — now bring back Flying Colonel status"*
   (a playful compliment) → *"I'm sorry you're missing Flying Colonel status…"*;
   *"can you show the WAS-DAL game on flight 74?"* (a request) → *"Sorry for the
   inconvenience…"*. *Hypothesis:* the draft prompt primes an apologetic frame;
   sentiment/act classification should gate the opening line.

5. **"DM your confirmation number" reflex even when it adds nothing.** For vague
   vents (*"why is it ALWAYS something 👎🏿"*) and answerable policy questions the
   draft still asks for a PNR, which drags helpfulness down (agent helpfulness
   3.82 vs tone 4.49). The judge also over-flags this reflex as "hallucination",
   inflating the rate in §3c. *Hypothesis:* the agent has one move; it needs an
   explicit "answer directly" branch for low-risk informational intents.

## 6. What is misleading about my headline number?

- **"Agent classification accuracy 0.83" hides that routing — the part that
  actually gates automation — is barely above a one-line rule (0.59 vs 0.65) and
  auto-sends 38% of should-escalates.** The system is not deployable on that
  number.
- **Routing accuracy is partly self-graded.** The router's rules and the golden
  routing labels were written from the same rubric by the same person. The
  honest signals are false-auto-rate and the independent baselines, not accuracy.
- **"DM your confirmation number" = escalate is my labelling call.** Flip it and
  ~35 golden labels move; the false-auto rate and "human load removed" figure
  change materially.
- **Single labeller** for both the golden set and the judge-validation human
  scores — there is no inter-annotator κ. 14 of the 100 eval rows are flagged
  `ambiguous`; on the `easy` slice routing is only marginally better (0.63).
- **Reference reply ≠ ground truth.** Delta's actual reply is often itself a
  templated "DM us"; scoring groundedness/helpfulness against it partly rewards
  imitating a deflection, and the verbatim-NN baseline looks artificially strong
  on groundedness (4.03) for the same reason.
- **The judge is a 27B open model with pooled κ 0.49 vs a human.** The reply-
  quality table can carry aggregate weight but not per-example claims, and the
  hallucination rate is inflated by the judge flagging the standard DM-ask.
- **100 examples, English-only, opening-message-only, 2017 data.** Confidence
  intervals on a 0.59 routing accuracy over n=100 are roughly ±10 points; none of
  this transfers to live multilingual multi-turn traffic.
- **Two draft models (gpt-oss-120b for 72, -20b for 28)** after the token cap —
  a small confound in the reply-quality numbers.

## 7. With one more week

- **Fix routing first.** Default medium-risk intents to escalate; replace keyword
  regexes with a small learned classifier over the signals; learn the confidence
  / similarity thresholds on a dev split instead of hand-setting them; add an
  abstain band routed to a lightweight human check.
- **Enforce grounding.** Post-generation check that every number / proper noun /
  policy claim in the draft appears in the retrieved context; strip agent
  signatures and URLs from precedent before it reaches the drafter.
- **Second annotator** on the golden set and the judge sample → real κ; adjudicate
  the 28 ambiguous rows.
- **Retrieval upgrade:** restrict precedent to same predicted intent, drop
  handoff-only replies, add a cross-encoder reranker.
- **Full-set eval** once off the free tier; expand golden to 250 with harder
  negatives (sarcasm, near-duplicate intents).
- **Adversarial safety pass:** prompt injection in tweets, fake compensation
  claims, impersonation.
- **Transfer test:** run the same pipeline on SouthwestAir with no code changes.
