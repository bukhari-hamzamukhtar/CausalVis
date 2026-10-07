# Where to submit (checked against the venues' own pages, 7 October 2026)

Publication and travel costs cannot be paid, so cost decides the order. One venue at a time.
An arXiv preprint is allowed alongside every option below, and both CVPR and TMLR say so in
writing.

## The two real candidates

### CVPR 2027 (Seattle, 19-26 June 2027)
Verified on cvpr.thecvf.com, 7 October 2026.

| item | fact |
|---|---|
| paper registration | 10 November 2026 AOE, fixed, no extension |
| full submission | 16 November 2026 AOE |
| supplementary material | 23 November 2026 AOE |
| reviews out / decisions | 25 January 2027 / 25 February 2027 |
| last cycle | 16,092 submissions, 4,089 accepted (25.4%) |
| length | 8 pages excluding references; over that is rejected without review |
| format | official template required; wrong template is rejected without review |
| anonymity | double blind; the public codebase cannot be cited, only promised |
| arXiv preprint | explicitly allowed, and so is naming the submission in a PhD application |
| dual submission | nothing substantially similar elsewhere, 16 Nov 2026 to 22 Feb 2027 |
| reviewer service | every author commits to review when invited |
| **cost if accepted** | **author registration is mandatory and virtual does not count: $525 early or $675 standard at the student member rate, plus IEEE/CVF membership** |

The last row decides everything. A virtual registration does not cover a paper, so an accepted
CVPR paper costs at least $525 (about 150,000 PKR) or it is withdrawn.

### TMLR (Transactions on Machine Learning Research)
Verified on jmlr.org/tmlr, 7 October 2026.

- Free to submit and free to publish. No registration, no travel, no article charge.
- Rolling submission, action editor assigned within a week, about two months to a decision.
- Acceptance asks two questions: are the claims supported by the evidence, and would some of
  its audience be interested. "Not novel enough" is not grounds for rejection.
- Scope names "new approaches for analysis, visualization, and understanding of learning
  systems" and reproducibility studies, which is what this paper is.
- Double blind, submitted anonymised to OpenReview, same as CVPR.
- No page limit, so the supplement can be part of the paper.
- Accepts only original work with no reuse of the authors' own prior papers. This paper
  qualifies.

## LLM policy at both, which is the question that prompted this check
Neither venue requires a disclosure statement. Both place responsibility on the author.

- CVPR 2026 FAQ: "Authors may use any tools they find productive in preparing a paper, but must
  be aware that they are responsible for any misrepresentation, factual inaccuracy, or
  plagiarism... It is not a defense to a charge of plagiarism or of inaccuracy to argue that
  'an LLM did it'." Papers citing non-existent material can be desk-rejected. CVPR 2027 says
  its own LLM section is still being finalised and will be updated before the deadline, so
  check it again in November.
- TMLR: "LLMs may be used as general-purpose assistive tools. Authors are fully responsible for
  content on which they are listed as (co-) authors."
- Both forbid prompt injection, meaning hidden text aimed at reviewers or tools. CVPR calls it
  an ethics violation and desk-rejects it.

So the one-line assistance statement is optional at both. It stays in the arXiv and journal
version and disappears automatically in the anonymous build, because acknowledgements break
anonymity.

## Recommended order
1. **arXiv** now. Free, and both venues allow it.
2. **TMLR**, unless someone else will pay a CVPR registration. Free at every step, the review
   criteria fit a measurement paper, and a decision takes about two months.
3. **CVPR 2027** only with funding in hand before 16 November. The question to ask is not
   whether Hamza can pay but whether GIKI will: universities routinely cover registration for
   an accepted paper at a top venue, and a CVPR paper from the department is worth that to
   them. Ask the supervisor, in writing, before committing.
4. **DMLR** if TMLR rejects. Free, and benchmark analysis is its core topic.
5. **A workshop** at CVPR, ICLR or NeurIPS. Most are non-archival, so a journal submission can
   follow. Only pay if you choose to attend.

Submitting to CVPR blocks TMLR from 16 November to 22 February. Submitting to TMLR first does
not block a later CVPR submission once TMLR decides, but an accepted TMLR paper is published,
so CVPR would then be out.

## What the recent CVPR award papers look like, for calibration
CVPR 2026 best paper D4RT (Google DeepMind, UCL, Oxford): a transformer that reconstructs
dynamic 4D scenes. Best student paper, structured latents for 3D generation (Tsinghua,
Microsoft). Honourable mentions: NitroGen (NVIDIA and others, a gaming foundation model trained
on 40,000 hours) and SAM 3D (Meta). The one exception to the pattern is the best student paper
honourable mention, ChordEdit, which is training-free, inversion-free and lightweight, from a
group with no large compute.

Four of the five are large systems from industry labs. None is a benchmark audit. That is not a
reason to avoid CVPR, since audits are accepted there under "datasets and benchmarks" and
"trustworthy vision", but it says what wins attention and it says the paper must read as a
measurement with a finding rather than as a system that scores below the leaders.

## Not worth it for this paper
- Journals with an article processing charge.
- Any venue that requires travel paid by the author.
- National conferences with a registration fee.

## Sources
- CVPR 2027 dates and call: https://cvpr.thecvf.com/Conferences/2027/Dates and /CallForPapers
- CVPR author guidelines and FAQ: https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines
- CVPR registration rates: https://cvpr.thecvf.com/Conferences/2026/Pricing2
- CVPR author kit: https://github.com/cvpr-org/author-kit
- TMLR: https://jmlr.org/tmlr/acceptance-criteria.html and /editorial-policies.html
