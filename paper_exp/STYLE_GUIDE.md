# Writing style for the CausalVis paper

Measured on 19 recent papers close to our topic (ICML 2025, ICLR 2025, NeurIPS 2022 to 2023,
TMLR 2023 and 2025, CVPR/WACV/ECCV 2024 to 2026, Nature Communications 2026, and arXiv 2025
to 2026), 176,662 words and 7,951 sentences in total:
Kang et al. (2025), Motamed et al. (2026), Krojer et al. (2025), Ishay et al. (2024),
Garrido et al. (2025), Bordes et al. (2025), Assran et al. (2025), Chow et al. (2025),
Meng et al. (2025), Bansal et al. (2025), Liu et al. (2024, PhysGen), Jin et al. (2023,
CLadder), Melnik et al. (2023), Cherian et al. (2024), Mao et al. (2022), Sharma and Fink
(2026), Ramachandran et al. (2025), Rädsch et al. (2026, Physics-IQ Verified) and
Thozhiyoor et al. (2025, sub-Earth gravity). The texts were read from arXiv HTML and counted
with short Python scripts (session scratchpad); each count below is over all 19 papers.

## 1. Numbers from the corpus

| feature | corpus value | our rule |
|---|---|---|
| median sentence length | 20 words (mean 22) | most sentences 12 to 30 words |
| passive sentences | 21% | about one in five, mostly in methods and data |
| "we" | 15.5 per 1,000 words | use "we" freely |
| em dash | 0.8 per 1,000 words; 6 of 19 papers use none | none |
| en dash | 0.6 per 1,000 words | only in number ranges written as "to" instead |
| semicolon | 3.5 per 1,000 words | allowed, sparingly |
| colon | 5.2 per 1,000 words | allowed, before lists and definitions |
| parentheses | 17 per 1,000 words | citations, numbers, abbreviations |
| "e.g." / "i.e." | 1.5 / 0.5 per 1,000 words | inside parentheses, followed by a comma |
| "points" for a difference | 21 uses; "percentage points" 3; "pp" 0 | "1.0 points" |
| "statistically significant", CIs | rare (1 and 6 uses) | we report paired z and 95% CIs, explained once |

Hyphens inside compound words are normal in every paper (state-of-the-art, per-option,
model-based, detection-only). Dashes used as punctuation are what we avoid.

## 2. Tense and voice
- Present tense for what the paper says and shows: "We present", "We find that", "Table 2 shows".
- Past tense for things done once in the study: "We trained", "we conducted", "we observed".
- Passive for data and procedures when the actor does not matter: "Examples are curated from nine sources."

## 3. Words the corpus uses (use these)
- After "we": use, find, provide, evaluate, present, compare, introduce, train, observe,
  propose, show, consider, report, focus, conduct, study, investigate, demonstrate, describe.
- For results: show, find, achieve, improve, observe, demonstrate, remain, struggle, fail,
  suggest, indicate, reveal, outperform, highlight, confirm, drop.
- Hedges: may, likely, suggests, indicates, largely, slightly, substantially, consistently.
  "Significant" only for a tested difference.
- Sentence openers, most common first: The, We, In, This, For, To, Our, As, However, These,
  While, It, Specifically, Additionally, Finally.
- Transitions, most common first: However, While, Specifically, Additionally, For example,
  Finally, In contrast, Furthermore, First, For instance, Therefore, To address, Notably, Thus,
  Moreover, Although, Overall, In addition, Unlike, Instead, In particular, Similarly.
- Frequent terms in this area: physical reasoning, world model, physical laws, counterfactual
  questions, state-of-the-art, ground truth, validation set, test set, question answering.
- Absent from the corpus, so we do not use them: delve, tapestry, intricate, "it is worth
  noting", "plays a pivotal role".

## 4. Two habits we never use
- Dashes as punctuation. Use a comma, a colon, parentheses or a new sentence.
- Restating a claim by denying its opposite ("X, not Y", "it is authentic, not fake").
  Say the claim once.

## 5. How the papers are built
**Abstract** (150 to 250 words). Context sentence. The problem, usually with "However".
"In this work, we ..." or "This paper introduces ...". How. Two or three numbers. One closing
line of implication ("Our study suggests that ...", "This highlights ..."). Physics-IQ Verified
(2026) opens its method sentence with "we present a systematic audit of", which is the closest
model for us.

**Introduction.** Context with citations, the gap, what we do, the main findings with
numbers, then a list introduced by one of: "In short, we make the following contributions:"
(Krojer et al., 2025, TMLR), "We summarize the main contributions of our work:" (Jin et al.,
2023), or ordinal sentences "First, ... Second, ... Finally, ..." (Ishay et al., 2024).

**Related work.** Grouped by topic. Each group opens with a short paragraph title that ends
with a period, for example "Neuro-symbolic video reasoning."

**Headings.** Numbered, sentence case: "1 Introduction", "2 Related work", "4.1 Setup".

**Citations.** TMLR uses author and year. Parenthetical: (Yi et al., 2020). Narrative:
Ishay et al. (2024) show that ... Several: (Ding et al., 2021; Chen et al., 2022).

**Figures and tables.** 91% of captions end with a period and 75% start with a short title.
Form: "Figure 3: Score against physics error. Each point is one simulator; ..." Tables have
the caption above: "Table 2: Counterfactual accuracy on CLEVRER. We report ...". In the text:
Figure 3, Table 2, Section 4, Appendix B (full words).

**Limitations.** A section or a titled paragraph: "Our work has several limitations. First,
... Second, ... Finally, ..." Krojer et al. (2025) open with "No benchmark comes without
limitations."

**Results sentences.** Number with its unit, then the comparison: "CausalVis reaches 90.4%
per option, 1.0 points below the fitted laws (z = 5.0)."
