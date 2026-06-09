<!-- Deferred implementation plan. Generated 2026-06-08. Execute when ready; do not auto-run. -->
# Plan: Slim CLAUDE.md below the 40 KB limit via path-scoped rules + dedup

> **STATUS: NOT YET IMPLEMENTED.** This is a saved plan for later execution (analysis was running
> in another process when it was written). When you pick this up, follow the *Implementation order*
> below, then the *Verification* checklist. Nothing in `CLAUDE.md`, `.claude/rules/`, or `TODO.md`
> has been changed yet.

## Context

`CLAUDE.md` is **43,512 chars / 274 lines** — over both the 40 KB limit and Claude Code's
recommended **≤200-line** target for memory files. Oversized memory files cost context every
session *and* measurably reduce instruction adherence. Much of the bloat is (a) deep solver/
walker internals that are only relevant when editing `decision_model.py`, and (b) long-form
write-ups that are **already duplicated** in subdirectory docs (`basin_estimation/findings.md`,
`VM_bifurcation_old_dtheta/VERDICT.md`, `weighting_analysis/README.md`).

Verified loading mechanics (from code.claude.com/docs/en/memory):
- `CLAUDE.md` loads **in full every session** — length directly costs context + adherence.
- **`@`-imports do NOT help** — imported files load at launch; pure cosmetic split.
- **`.claude/rules/*.md` with a `paths:` frontmatter load ONLY when Claude reads a matching
  file** — the real on-demand mechanism. Rules *without* `paths:` load every session (no savings).
- Pull-only docs (plain `.md` referenced by a pointer) load only when Claude chooses to Read them.

**Decisions (confirmed with user):**
- **Mechanism = Hybrid.** Path-scoped `.claude/rules/` for content tied to a code area; fold the
  genuinely-duplicated write-ups down to one-line pointers at the existing pull-only docs.
- **Trim depth = Aggressive (~130 lines).** Keep only the durable core + a reference index inline.

**Outcome:** `CLAUDE.md` drops to ~130 lines / well under 40 KB, with zero information lost — every
moved section lands in a named destination that auto-surfaces when relevant.

## Non-goals / out of scope
- **No code changes.** `decision_model.py` and all scripts are untouched.
- **No commit/push** (project git policy — explicit ask required each time).
- **No `@`-imports added** (they wouldn't reduce context).
- Existing exhaustive docs (`findings.md`, `VERDICT.md`, weighting `README.md`,
  `theory_background.md`, `free_energy_derivation.md`) stay as-is — they're already pull-only.

## New file layout

```
.claude/
  rules/
    perception-and-solver.md   # paths: [decision_model.py]   — numerics
    walker-dynamics.md         # paths: [decision_model.py]   — plot_walkers / noise
    torque-and-stability.md    # paths: [decision_model.py]   — dθ/dt + Jacobian theory
    basin-estimation.md        # paths: [basin_estimation/**]  — thin: facts + →findings.md
    bifurcation-explorations.md# paths: [VM_bifurcations/**, bifurc_plots/**] — thin: →VERDICT.md
    weighting.md               # paths: [weighting_analysis/**] — thin: →README.md
CLAUDE.md                      # slimmed to ~130 lines + reference index
TODO_Jan_2026.md               # RENAMED from TODO.md (archived stale research notes)
TODO.md                        # NEW — Claude engineering TODO (moved "Open TODOs")
```

Each rule begins with YAML frontmatter, e.g.:
```markdown
---
paths:
  - "decision_model.py"
---
```
The three `decision_model.py`-scoped rules auto-load together whenever that file is opened —
i.e. exactly the sessions that need them — and cost **nothing** on planning / subdir / doc sessions.

## Section-by-section disposition

Source line ranges are from the current `CLAUDE.md` (`## ` headers via grep).

| CLAUDE.md section (src lines) | Action | Destination |
|---|---|---|
| Title + intro (1–3) | keep (trim 1 line) | CLAUDE.md |
| Git policy (5–7) | **keep verbatim** — load-bearing every session | CLAUDE.md |
| Codebase layout (9–20) | keep, light trim | CLAUDE.md |
| Coordinate systems (22–30) | keep — core concept | CLAUDE.md |
| Two model classes — NBM/IEM overview + key methods (32–59) | keep trimmed (overviews + method links + load-bearing warnings) | CLAUDE.md |
| PerceptionModel API deep detail (61–74: kwargs/`_FAMILY_INFO`/Old→new table) | **move**; keep only "two roles + assign `a_warp`/`b_warp`" summary inline | `perception-and-solver.md` |
| Self-consistent equilibria (76–89) | keep conclusion; **move** the "three approaches considered" history | CLAUDE.md (concl.) |
| Solver architecture (90–170) | **move almost all**; keep 4–5-line summary + the two preserve-this gotchas | `perception-and-solver.md` (numerics) + `walker-dynamics.md` (the `plot_walkers` subsections) |
| Stability criterion (171–193) | keep short summary; **move** the correctness proof + over-counting example | `torque-and-stability.md` |
| Half-angle torque dated note + K=2 derivation (inside 171–193) | **move** | `torque-and-stability.md` |
| Geometry: targets (194–201) | keep brief | CLAUDE.md |
| Bifurcation diagram conventions (202–210) | keep as terse guardrail bullets | CLAUDE.md |
| Physical phenomena (211–246) | **trim to one-line pointers** (decision-paralysis, Hopf island→VERDICT.md, ears→weighting README) | CLAUDE.md pointers + existing docs |
| Basin estimator (247–259) | trim to status + pointer; working summary | CLAUDE.md (pointer) + `basin-estimation.md` (facts) + `findings.md` |
| Open TODOs (260–268) | **move** engineering items to fresh `TODO.md`; keep 1-line pointer | new `TODO.md` |
| Common gotchas (269–274) | **keep** — these are the always-do rules | CLAUDE.md |
| — | **add** Reference index table | CLAUDE.md |

### Rule contents (what moves where)
- **`perception-and-solver.md`** ← PerceptionModel deep API (kwargs, `_FAMILY_INFO`, Old→new table,
  spline construction), perception interval arithmetic, integral-spline accuracy/perf notes, NBM
  `sc_equilib` single-pass, IEM `sc_equilib` multistart, `run_dgamma_dt` LSODA reformulation, the
  `R<0.01` filter, residual-asymmetry note.
- **`walker-dynamics.md`** ← the `plot_walkers` subsections: Euler-Maruyama state-gated noise law,
  `walk_std` blind-spot search, two-layer target detection, `R_exp` drift exponent, the three loss
  mechanisms / strong-coupling orbit note, degenerate-histogram guard.
- **`torque-and-stability.md`** ← half-angle heading-torque dated note (2026-06-02), `K=2`
  invariance derivation, 3×3 coupled-Jacobian correctness proof, `discrim_a` legacy / over-counting
  example. (Points to `VM_bifurcation_old_dtheta/VERDICT.md` for the θ-side recalibration caveat.)
- **`basin-estimation.md`** (thin) ← status (vetted, not wired in), public entry point
  `compute_basins_at_focal_loc`, the load-bearing facts (ΔF_γ only meaningful in multistable cells;
  4 boundary kinds incl. branch-cut; ~4.2 min/41×41 cost), then "full detail → `findings.md`".
- **`bifurcation-explorations.md`** (thin) ← orient to `bifurc_plots/` script roles + "→ VERDICT.md".
- **`weighting.md`** (thin) ← the "ears" one-liner + uniform-default note + "→ `weighting_analysis/README.md`".

## Target CLAUDE.md skeleton (~130 lines)

```
# Neural Ring Model — Project Guide               (intro, 3 lines)
## ⚠️ Git policy (do not deviate)                  (verbatim)
## Codebase layout                                 (trimmed map)
## Coordinate systems                              (kept)
## Two model classes
   ### NeuralBandModel (NBM)                        (overview + key-method links)
   ### IsingExtModel (IEM)                          (overview + key-method links)
   ### PerceptionModel — warp/weight decoupled      (two-roles summary + IEM-identity warning
                                                      + "assign a_warp/b_warp"; detail → rule)
## Self-consistent equilibria                       (conclusion only)
## Stability criterion                              (default = 3×3 coupled; methods; detail → rule)
## Geometry: targets                                (3 geometries, brief)
## Bifurcation diagrams — conventions               (guardrail bullets)
## Physical phenomena                               (one-line pointers each)
## Basin-of-attraction estimator                    (status + pointer)
## Open TODOs                                        (pointer to TODO.md)
## Common gotchas                                    (kept)
## Detailed references (auto-loaded rules + docs)    (NEW index table)
```

Reference-index table maps each topic → "auto-loads when" → full-detail file, so a human reading
CLAUDE.md can see where everything went and Claude knows which rule fires when.

## TODO.md handling (rename + fresh file)
The current `TODO.md` is forward-looking **research-direction prose** (authored narrative — JWB/AJB
notes, "Christopher calls dibs", scale-up thoughts) and is **stale**. It is distinct in kind from
CLAUDE.md's "Open TODOs" (engineering status: IEM LSODA port, cell-center sampling, basin wiring,
foveal-weight analysis, residual-noise-floor idea).

- **Rename** `TODO.md` → `TODO_Jan_2026.md` (use `git mv` to preserve history; no commit). This
  archives the dated research notes under a clear name.
- **Create a fresh `TODO.md`** = the **Claude engineering TODO**, populated with the engineering
  "Open TODOs" moved out of CLAUDE.md (IEM `run_dgamma_dt` LSODA port; cell-center sampling for
  bifurcation refinement; blind-spot RESOLVED note; residual heading-noise floor idea; foveal
  `angle_weight` commitment-signal analysis; two-panel bifurcation+basin plot; dynamical §9 basin
  ratio recompute).
- CLAUDE.md's "Open TODOs" section collapses to a one-line pointer to `TODO.md`.

## Implementation order
> Deferred — analysis is running in another process. This plan is persisted to the repo base
> directory for execution later; do NOT run it until ready.
1. Re-read current `CLAUDE.md` (it was touched by user/linter) to get exact current text.
2. Create the six `.claude/rules/*.md` files (frontmatter + moved content, verbatim where possible).
3. `git mv TODO.md TODO_Jan_2026.md` (preserve history; no commit), then create a fresh `TODO.md`
   holding the engineering "Open TODOs" moved out of CLAUDE.md.
4. Rewrite `CLAUDE.md` to the slim skeleton + reference index (incl. pointer to new `TODO.md`).

## Verification
- `wc -c CLAUDE.md` → **< 40000** (target ~22–26 KB); `wc -l CLAUDE.md` → ~130.
- `grep -c '^@' CLAUDE.md` → 0 (no eager imports introduced).
- Each rule file: `head -5` shows valid `paths:` YAML frontmatter; `find .claude/rules -name '*.md'`
  lists all six.
- **No-loss check:** grep the moved keywords to confirm each landed somewhere — e.g.
  `LSODA`, `interval arithmetic`, `walk_std`, `half-angle`, `coupled-Jacobian`/`over-count`,
  `compute_basins_at_focal_loc`, each "Open TODO" item — should appear in a rule or TODO.md.
- **Load-bearing gotchas still present in CLAUDE.md:** git policy; "preserve exact interval
  arithmetic"; "preserve real-valued LSODA reformulation"; "assign `a_warp`/`b_warp`"; "IEM must use
  identity warp". (Keep the *directive* inline even where the *explanation* moved.)
- After implementation, run `/memory` to confirm the six rules + CLAUDE.md are listed as loaded, and
  optionally open a file under `basin_estimation/` to confirm the scoped rule fires.
- **TODO rename check:** `TODO_Jan_2026.md` exists with the old research-direction prose; new
  `TODO.md` holds only the engineering items; `git status` shows the rename (R) + new file.
- This is docs-only — no tests to run; `git status` should show only `.md` files changed.
