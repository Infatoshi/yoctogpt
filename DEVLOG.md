# DEVLOG

The journey of building yoctogpt into a course. Two halves: the agency/momentum lessons (how to actually finish), and the taste/design lessons (how to make it good). Written so future-me, on the next course, doesn't relearn this the hard way.

---

## Part 1 — Agency: actually shipping the thing

### The perfectionism wall is built one defensible brick at a time
The arc of this project: working notebook → polish the diagrams → re-do the diagrams → add captions → rewrite the whole notebook (v1→v2) → rewrite the autograd (topo-sort→tape) → consolidate files → nearly build a Manim pipeline. Every single step was defensible in isolation. Cumulatively they were a wall, and a full week of zero recording happened right after the biggest one (the v2 rewrite that invalidated ~70 min of footage). The lesson isn't "don't improve things." It's that improvement work *feels* like progress while the actual deliverable (a recorded video) doesn't move an inch. Polishing is the most comfortable way to not ship.

### Recording was always the bottleneck. Never the assets.
Diagrams, notebook structure, render pipelines — none of it is the constraint. The constraint is one person being willing to talk into a mic. Time spent on anything else is, at best, sharpening the axe and, at worst, avoidance dressed as craft. Default suspicion: if I'm building instead of recording, ask why.

### Record first, polish only what proves weak on camera
The right order is: record the whole thing with the assets you already have, *feel* which 2-3 moments fall flat, then fix only those. Guessing in advance what needs a fancy animation or a perfect diagram produces ten polished things you half-use. The Manim question crystallized this — the answer was "don't build the pipeline; record, and animate only the 2-3 inherently-temporal concepts if they actually fail." Earn the polish by knowing precisely what it must fix.

### Deletion is a momentum tool
Twice this project, the move that unstuck things was deleting footage — "delete everything after 7," then down to 3 takes. A clean slate beats a guilt pile of half-good takes you keep meaning to salvage. Sunk-cost footage is a tax on every future decision ("but can I reuse take 12's middle section?"). Burning it is freeing. (Trash, not hard-delete — recoverable for a month — but psychologically: gone.)

### The struggle takes were not wasted
~69 minutes of footage got cut because it showed code that got rewritten, plus the takes where the explanation flailed. But those flails are *why* the explanations got good: the tape-autograd framing, the one-neuron blame diagram, the softmax max-subtraction "overflow doorman," the "every op melts into a coefficient" line — all came from watching the first attempts not land. First takes are reconnaissance. You can't think your way to the clean explanation; you have to hear the messy one out loud first.

### "Don't animate what you can show live"
The single best "animation" in the course is `python3 v1.py` streaming gibberish into "Peter Piper" in a real terminal. No Manim clip beats a real thing happening. Reach for synthetic production value only where reality can't show it.

### The helper gilds too
Worth naming: the AI editor/advisor in this loop also kept proposing improvements (re-diagram, re-rewrite, dark mode, manifests). Enthusiasm for tractable, visible-output work is itself part of the trap, on both sides of the keyboard. "Is this moving us toward a recorded video?" is the only question that matters, and someone has to keep asking it.

### When momentum is gone, shrink the ask
Coming back after a week, the useful move was never "here's the 70-minute re-record list." It was "record the cold open, that's it, then come back." One section, dropped in, transcribed, done. Activation energy is the whole game; lower it ruthlessly.

---

## Part 2 — Taste & design: making it good

### The notebook is a course *companion*, not a standalone handoff
Once there's a voice track, prose in the notebook competes with the narration. The notebook should carry: section headers (Ctrl-F anchors), formulas, worked numbers, diagrams, and one-line intuitions to pull from. Everything else is said out loud. Walls of markdown are a smell — they're a handoff doc's job, not a companion's.

### Teach from readable code; reveal the golfed version as a trophy
`v1.py` is 97 dense lines (dict weights, comprehensions, one-line backward). That's the *trophy* shown at the end — "everything you learned, on one screen." You never teach from it. The notebook teaches with named variables, expanded loops, one inline layer. Beginners read the call sites constantly and the clever compression once. (Corollary: the dict-vs-named divergence between trophy and course is fine — the trophy is *allowed* to be compressed.)

### Know your one audience: basic Python, zero ML
Not "the general public" (they don't write Python) and not "people who know ML" (they don't need this). That single reader dictates everything: never explain a for-loop, never assume a gradient. Violating either end loses them.

### Calculus without the word "calculus"
Nudge, slope, leverage, coefficient. The manual-nudge cell (3.001² → slope 6) *is* a derivative with the scary word removed. Never say "derivative" or "chain rule" cold; build the intuition, then optionally name it.

### Motivation-first: show the payoff before the mechanism
Backprop only makes sense once you've seen the optimizer step (`w -= lr * grad`) actually move a weight toward the target. Show learning happen first, then say "the only mystery left is: who computed that gradient?" People can't care about a mechanism until they've seen what it's for. (Same principle as the cold open, applied to a single concept.)

### The big explanations we landed on (keep these)
- **Backprop = a guest list + one line of arithmetic.** Build the tape at birth; walk it backwards; `child.grad += local_slope * parent.grad`.
- **Tape over topological sort.** "A result is always younger than its ingredients, so read the birth list backwards." Kills recursion/visited/DFS, and it's literally what PyTorch calls the tape.
- **Every op melts into a coefficient at the current values.** Squaring at 6 is just "×12," ReLU is "×1 or ×0." Backprop walks a graph of plain coefficients: multiply along a path, add across paths. The slope is *minted at forward time and frozen*; backward just spends it.
- **Each input's leverage is the OTHER input's value** (the `__mul__` local-grad tuple). x.grad uses w.*data*, never w.grad. Two gradients never multiply.
- **softmax max-subtraction is overflow protection, and it's free** (the shift cancels in the division). Demo softmax([1,2,3]) == softmax([101,102,103]).
- **Init: no magic number, only a safe band.** Measured: zeros freeze at ln(24) forever, all-ones crashes on log(0), std 1.0 starts at loss 175, the band 0.005–0.3 all works. Discovered in 3 runs, not asserted.

### Discovery pacing (the micrograd move)
Cells you run by hand, repeatedly, watching the number change — one training step, shift-enter five times — *then* wrap in a loop. Same for generation: one character per run, then loop it. Stumbling into the abstraction with the viewer beats presenting it finished.

### Defer jargon until the code earns it
"logits" appears as a comment on the line that creates them, not three sentences earlier. Introduce a term at the moment of first use, never as front-loaded vocabulary. (The terminology glossary lives at the *bottom*, as an appendix, not the top.)

### Deliberate simplifications, stated as choices
One layer, one attention head, no batching, no biases, ReLU only, character tokenizer. Each is a thing you *say on camera* you're omitting and why ("batching is a throughput trick, the model is identical"), with a pointer to where it's done for real. Simplicity you narrate is pedagogy; simplicity you hide is a gap.

---

## Part 3 — Visual taste (diagrams)

The bar: **authentically human**. Subtle sloppiness reads as AI-generated and quietly destroys trust, especially full-screen in a video.

- **Edge-exact arrows.** Compute geometry; never eyeball. If arrows won't land centered, the *source nodes* are probably misaligned — fix those first (the chain-rule diagram had nodes 10px off-axis so no arrow could ever be right).
- **Curvy hand-drawn lines on everything.** Scale the bow to the line length, alternate bow direction so it looks organic, not patterned. (Started as "only long lines"; the rule became *all* lines.)
- **Numbered reading-order sections.** Circled badges + short lowercase labels + light flow arrows between them, so the layout teaches the *order*, not just the parts. Essential for any multi-step diagram (attention especially).
- **Titles demoted to bottom captions.** The mechanism is the hero; the title is a footnote. No exceptions, even formula titles.
- **Show the intermediate step.** The attention diagram failed until the query *visibly participated* — added an explicit score column (q·k → score → softmax → weight) instead of jumping from keys to percentages.
- **Low density.** Badges get their own space, ~1.6 line spacing, bigger boxes over smaller fonts. A wall of numbers in a box is as bad as a wall of prose.
- **Dark mode by default.** Light diagrams are a flashbang against a dark notebook/editor. (`exportWithDarkMode` + feed it a *white* background so the invert filter turns it near-black; feeding it dark inverts to light gray.)
- **Verify by zooming into the rendered PNG** at native res before declaring done. Overlaps and squished line-heights only show at 1:1.

---

## Part 4 — Production & tooling (hard-won)

- **Capture a 2560x1440 window, not the full ultrawide.** Full 21:9 either letterboxes to a thin strip or downscales code into mush. Pin the Chrome window to exactly 16:9.
- **Use the real mic.** Default input had silently been AirPods over Bluetooth at 24kHz (phone-call quality); the Yeti at 48kHz was sitting right there. Check input device before every session.
- **Camera:** FOV ~65° (default is too wide, makes you small), lens just above eyeline tilted down ~10°, PiP corner ~18% width. Full-frame only for the opening motivation beat.
- **Editing pipeline that works:** mlx-whisper (base.en) for transcripts; whisper *suppresses* fillers, recover them with an um/uh initial-prompt + small.en, cut >=0.6 confidence; compress silences >1.6s to ~0.7s keeping breath on each side; ffmpeg filter_complex trim/concat. Verify seams by transcribing ~12s around each join.
- **Record in any order; label after.** OBS timestamp-names takes; transcribe and rename to `N-description.mov` from content. The manifest (~/Movies/MANIFEST.md) tracks keep/re-record status.

---

## The one-line version

Build less, record more. The explanation you'll keep is the one you had to flail through first. Make it look human, show the payoff before the mechanism, and delete without mercy.
