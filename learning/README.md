# Learning materials

Everything written to **learn from** this project — three levels, read in this order:

1. **[Learning Guide](Quran-ARS-Learning-Guide.md)** — the *tour*. From audio → ML → the data →
   the models → the final system → grading → the lessons. No ML background assumed.
2. **[Deep-Dive series](deep-dive/)** — the *textbook*. Eight chapters with the real math,
   derivations, worked examples, and exercises (DSP/features, architectures & attention, CTC,
   seq2seq & decoding, self-supervised + LoRA, evaluation & alignment, phonetics & Arabic,
   pronunciation assessment), plus **[Appendix A](deep-dive/appendix-A-worked-examples.md)**
   (Viterbi/forced-alignment + attention backprop) and **[`from_scratch.py`](deep-dive/from_scratch.py)**
   — runnable pure-NumPy log-mel + CTC + attention with self-tests.
3. **[Technical Documentation](Quran-ARS-Technical-Documentation.md)** — the *exact record*:
   final architecture, every model evaluated, the full experiment log, cost/server analysis.

Start with the Learning Guide; do the deep-dive exercises and run `from_scratch.py` alongside
Chapters 1–3.
