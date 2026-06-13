# DocsRAG — Stress-Test Playbook

Drop all the files from `cricket_test_docs/` into your project's `docs/` folder,
re-run `python build_index.py`, then work through the questions below. Each document
is engineered to break a different part of a RAG system. The goal is not to "pass" —
it's to SEE the failure, understand why, and learn the fix. Bring the disappointing
answers back and we'll diagnose them together.

---

## The corpus and what each file attacks

**01_laws_manual.txt** — long, dense, numbered rules.
Failure mode: *retrieval depth & chunk boundaries.* Rules are terse and similar to
each other, so the right passage is easy to miss among near-duplicates.

**02_club_faq.md** — deliberately CONTRADICTS the other docs.
Failure mode: *conflicting sources.* It says teams have 8 players (social league)
while the laws manual says 11. Watch whether the model notices the conflict or just
picks one at random.

**03_player_stats_2024.md** — tables and numbers.
Failure mode: *tabular/numeric retrieval.* Embeddings are weak at "who has the
highest average" type questions because tables chunk badly and numbers don't embed
semantically.

**04_newsletter_two_column.pdf** — two-column layout.
Failure mode: *text-extraction order.* Column text often extracts interleaved, so
chunks become word-salad. This is the single most common real-client document problem.

**05_coaching_manual.pdf** — multi-page with repeated header/footer.
Failure mode: *noise pollution.* "CONFIDENTIAL — DO NOT DISTRIBUTE" and "Page N"
repeat on every page and leak into chunks, diluting relevance.

---

## Questions to ask (and what to watch for)

### A. Retrieval depth — should mostly work, sets your baseline
1. "What is the penalty for a no ball?"  → expect a correct, cited answer.
2. "How wide is the wicket?"  → tests retrieving one specific number from a sea of similar rules.

### B. Multi-part questions — exposes whether k=4 is enough
3. "What are all the ways a batter can be dismissed?"  → the answer is spread across
   Laws 32–39 (five+ separate chunks). With k=4 it will likely MISS some. This is your
   clearest "increase k / the answer is incomplete" lesson.

### C. Conflicting sources — exposes contradiction blindness
4. "How many players are on a team?"  → the laws say 11, the FAQ says 8. A good system
   flags both; a naive one silently picks one. Note which it does, and whether the cited
   source explains the discrepancy.
5. "How many overs can one bowler bowl?"  → FAQ gives 8 (40-over) and 2 (10-over) and
   mentions the ODI limit of 10. Watch if it conflates the three.

### D. Numbers & tables — exposes embedding's weakness with data
6. "Who was the leading run-scorer in 2024?"  → stated in plain text, should work.
7. "Who had the best bowling average?"  → requires reading the table; S. Khan (14.37).
   Watch if it confuses 'average' with 'wickets' or grabs the wrong row.
8. "How many matches did the team win?"  → simple lookup; tests if the summary chunk
   was retrieved over the table chunks.

### E. PDF extraction quality — the real-world killer
9. "Who was named Player of the Season?"  → appears in BOTH the stats doc and the
   two-column newsletter. Compare which source it cites; if the newsletter text is
   jumbled, it may retrieve the cleaner markdown instead, which actually masks the
   extraction problem. To SEE the problem directly, run step 10.
10. Inspect extraction yourself:
    ```python
    from ingest import load_documents
    for c in load_documents("docs"):
        if c["source"].endswith(".pdf"):
            print(c["source"], "#", c["position"]); print(c["text"][:200]); print("---")
    ```
    Look at the two-column newsletter chunks: is the text in reading order, or are the
    left and right columns interleaved into nonsense? That garbled text is what got
    embedded. THIS is the failure most beginners never notice.

### F. Honest "I don't know" — the trust test
11. "What was the team's sponsorship revenue?"  → not in any document. It MUST say it
    doesn't know. If it invents a number, your prompt/grounding needs tightening.

---

## After you've run these, the tuning levers we'll pull together
- **chunk_size / chunk_overlap** in `ingest.py` — for dense rules vs prose.
- **k** in retrieval — multi-part questions need more retrieved chunks.
- **PDF extraction** — when `pypdf` jumbles columns, switch extractor (e.g. pdfplumber)
  or add layout-aware parsing. This is a genuinely senior skill clients pay for.
- **header/footer stripping** — clean repeated noise before chunking.
- **table handling** — extract tables separately or convert rows to sentences before
  embedding.
- **prompt hardening** — make "I don't know" stickier to kill hallucinations.

Bring me the worst answer you get and we'll fix that one end-to-end first.
