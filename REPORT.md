# Delta Twitter Support Agent — Report

<!-- Numbers are filled from reports/results.json + reports/judge_agreement.json.
     Placeholders look like {{like_this}} until run_all.py has been run. -->

## 1. Problem framing

**Brand:** @Delta (US airline). **Task:** for an incoming customer tweet — (1)
classify intent, (2) draft a public reply grounded in how Delta has historically
answered similar tweets, (3) decide auto-send vs. escalate to a human, with a
reason.

**What "good" means here.** Delta's support Twitter is a *triage and holding*
channel, not a resolution channel — most substantive fixes happen in DMs with
account access. So a good agent:
- never auto-sends something that could mislead a customer or commit Delta to a
  policy/refund it wouldn't honour (safety first);
- handles the genuinely safe traffic without a human — compliments, general
  policy questions, feedback acknowledgement (that's most of the volume);
- for everything else, escalates *with a specific reason* a human can act on.

The metric that matters is **false-auto rate** (should-escalate messages that got
auto-sent) — kept near zero — followed by how much human load is removed.

**What I deliberately did not build:**
- No fine-tuning — few-shot + retrieval only.
- No multi-turn dialogue state — the agent sees the opening message only.
- No live integrations (flight status, PNR lookup, posting to Twitter).
- No non-English handling — filtered out (5% of Delta threads).
- No generation of the DM conversation that follows an escalation.

## 2. System

```
tweet ─▶ classify (gpt-oss-120b, 8 intents + other)
      ─▶ retrieve (FAISS / bge-small, top-5 past resolved Delta threads)
      ─▶ draft   (gpt-oss-120b, grounded ONLY in retrieved replies; may abstain)
      ─▶ route   (rule layer: safety/money/disruption/churn/live-data/intent-risk/
                  confidence  ▸ then LLM upgrade-only safety check)
```
Corpus = 6,000 older threads; golden set = 199 hand-labelled newer threads
(time-split, no leakage). Full details: `INTENTS.md`, `docs/golden_set_note.md`,
`DECISIONS.md`.

## 3. Results vs. baselines

### 3a. Intent classification

| system | accuracy | macro-F1 |
|--------|---------:|---------:|
| trivial — majority class | {{cls_triv_acc}} | {{cls_triv_f1}} |
| simple — kNN (leave-one-out) | {{cls_knn_acc}} | {{cls_knn_f1}} |
| **agent — gpt-oss-120b few-shot** | **{{cls_agent_acc}}** | **{{cls_agent_f1}}** |

Weakest intents: {{cls_weak_intents}}.

### 3b. Routing (auto vs. escalate)

| system | accuracy | escalate-recall | **false-auto** | over-escalation |
|--------|---------:|----------------:|---------------:|----------------:|
| trivial — always escalate | {{r_esc_acc}} | 1.00 | 0.00 | 1.00 |
| trivial — always auto | {{r_auto_acc}} | 0.00 | 1.00 | 0.00 |
| simple — escalate if intent-risk = high (pred intent) | {{r_rule_acc}} | {{r_rule_rec}} | {{r_rule_fa}} | {{r_rule_oe}} |
| **agent router** | **{{r_agent_acc}}** | **{{r_agent_rec}}** | **{{r_agent_fa}}** | **{{r_agent_oe}}** |

### 3c. Reply quality (LLM judge, 1–5)

| reply | overall | groundedness | helpfulness | tone | safety | hallucination |
|-------|--------:|-------------:|------------:|-----:|-------:|--------------:|
| trivial — canned holding reply | {{rep_canned}} | | | | | |
| simple — nearest past reply, verbatim | {{rep_nn}} | | | | | |
| **agent draft (all)** | **{{rep_agent}}** | | | | | {{rep_agent_hall}} |
| agent draft — auto-sent only | {{rep_auto}} | | | | | {{rep_auto_hall}} |

Agent draft vs. Delta's actual reply, cosine similarity: {{rep_cos}}.

## 4. Is the judge trustworthy?

40 agent replies scored by a human and by the judge (`qwen3.8-27b`) on the same
rubric.

| dimension | quad-weighted κ | Spearman ρ | mean abs error |
|-----------|---------------:|-----------:|---------------:|
| groundedness | {{j_g_k}} | {{j_g_s}} | {{j_g_m}} |
| helpfulness | {{j_h_k}} | {{j_h_s}} | {{j_h_m}} |
| tone | {{j_t_k}} | {{j_t_s}} | {{j_t_m}} |
| safety | {{j_s_k}} | {{j_s_s}} | {{j_s_m}} |
| **pooled** | **{{j_p_k}}** | **{{j_p_s}}** | **{{j_p_m}}** |

Where judge and human diverge: {{judge_divergence}}.

## 5. Top 5 failure modes

1. {{fail_1}}
2. {{fail_2}}
3. {{fail_3}}
4. {{fail_4}}
5. {{fail_5}}

## 6. What is misleading about my headline number?

- **Routing accuracy is partly self-graded.** The router's rules and the golden
  routing labels were written from the same rubric by the same person. Read
  escalate-recall / false-auto instead, and the independent baselines.
- **"DM your confirmation number" = escalate** is my call. Flip it and ~35 labels
  move; the agent's escalation rate and the "human load removed" figure change
  materially.
- **Single labeller** for the golden set and the judge-validation human scores —
  no inter-annotator agreement number. Ambiguous rows (28/199) are where this
  bites.
- **Reference reply ≠ ground truth.** Delta's actual reply is often itself a
  templated "DM us"; scoring groundedness/helpfulness against it rewards
  imitating a deflection.
- **English-only, opening-message-only, time-boxed to 2017 data** — the numbers
  don't transfer to live multilingual multi-turn traffic.
- **Judge is a 27B open model**, not frontier; its κ with a human caps how much
  weight the reply-quality table can bear.
- **checkin_boarding F1 is over 7 examples** — noise.

## 7. With one more week

- Second annotator on the golden set + judge sample → real κ, and adjudicate the
  28 ambiguous rows.
- Confidence-calibrated routing: learn thresholds on a dev split instead of hand-
  set floors; add an abstain band that routes to a lightweight human check.
- Retrieval upgrade: filter precedent to same-intent, drop handoff-only replies,
  try a cross-encoder reranker.
- Expand golden to 250 with harder negatives (near-duplicate intents, sarcasm).
- Add a second brand (SouthwestAir) to test whether the pipeline transfers.
- Adversarial safety pass: prompt-injection in tweets, fake compensation claims.
