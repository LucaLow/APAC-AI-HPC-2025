# BID '26 Submission Checklist — deadline July 15, 2026, 23:59 AoE

(AoE = July 16, ~9:59 PM AEST. Do not use that buffer. Submit by the 14th.)

## Requirements (all currently met by main.pdf)
- [x] ACM Primary Article Template (acmart/sigconf), single blind = authors listed
- [x] Max 8 pages including references — we're at 6
- [x] English
- [ ] Submit at https://easychair.org/conferences/?conf=bid2026

## Before submitting — Luca must do
1. **Nathan's surname + both emails** — title block currently says "Nathan SURNAME" and
   example.com emails. Fix in main.tex lines ~18–30, recompile (or tell Claude).
2. **Read the paper once.** Especially §4 (Results) — every number is log-verified
   (see NUMBERS.md for the number → log-file mapping), but the framing is new
   relative to the tech report.
3. **Decide on the two flagged items:**
   - CUDA 12.6 baseline (5,839 tok/s): appears in the tech report but no log exists in
     this folder. Either locate the log, or we soften/remove that comparison.
   - "We release our build scripts, job scripts, and raw benchmark logs" (end of §1):
     needs a public repo (GitHub) before camera-ready, or delete the sentence.
4. **Acknowledgments**: currently thanks NSCC + "the operators of the H200 system"
   anonymously. Name Firmus explicitly? Check any usage/publicity terms first.
5. Optional: artifact appendix (doesn't count toward page limit) — scripts + logs
   would strengthen it; can be added for camera-ready.

## EasyChair steps (10 min)
1. Log in / create account at the link above → "make a new submission"
2. Authors: both names, emails, affiliation (Monash DeepNeuron, Monash University),
   country Australia, mark Luca as corresponding
3. Title: When One Node Beats Two: Benchmarking DeepSeek-R1 Inference with SGLang
   on H100 and H200 Systems
4. Abstract: paste from main.tex (plain text, strip LaTeX escapes)
5. Keywords: LLM inference; benchmarking; SGLang; DeepSeek-R1; parallelism; NCCL;
   GPU clusters (one per line)
6. Upload main.pdf → Submit → confirm the confirmation email arrives

## Dates
- Notification: July 20, 2026
- Camera-ready: July 30, 2026 (ACM rights form + any reviewer fixes)
- Workshop: Sept 28, 2026, NTU@One-North, Singapore (remote presentation supported)

## Housekeeping
- Delete the stray `ziriDjeP` file in this folder (zip temp; Claude lacks delete permission)
- To recompile after edits: pdflatex main → bibtex main → pdflatex ×2,
  or upload latex_source_for_overleaf.zip to Overleaf (compiles as-is)
