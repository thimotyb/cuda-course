---
name: material-authoring
description: Author and update CUDA course modules with syllabus alignment, clickable figures, and content locking.
---

# CUDA Material Authoring Skill

Use this skill when writing or revising module content for `cuda-course`.

## Source Priority

- Treat `resources/sillabo.txt` as the official syllabus when present.
- If `resources/sillabo.txt` is missing, use `resources/sillabo-corso.txt` as the current syllabus copy and report the filename mismatch.
- Use `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` as the primary technical source.
- Use official NVIDIA CUDA documentation, CUDA Best Practices, PyTorch CUDA docs, and curated course links as supplemental sources.
- Keep the site module structure aligned 1:1 with the syllabus modules already mapped in `course-structure.md`.

## Required Content Structure

- Work incrementally, one module at a time.
- Use exactly two heading levels inside module content: section `x` as `h2` and subsection `x.y` as `h3`.
- Number all content sections and subsections explicitly.
- Render each subsection in its own rectangular slide-like block, using the existing `.subtopic` pattern unless the site design evolves.
- Keep terminal module sections for guided exercises and summary material consistent with the local non-regression guard.

## Required Navigation Behavior

- Keep a left navigation sidebar on every module page.
- Sidebar must be collapsible.
- Every sidebar item must be clickable and navigate to the matching section or subsection.
- Sidebar structure must match the page `h2`/`h3` structure 1:1.

## Required Figure Behavior

- Add figures where they improve conceptual understanding, especially for architecture, memory hierarchy, execution mapping, data movement, and performance workflows.
- Number every figure with the module id and sequence, for example `Figure M1.1`.
- Every figure must be clickable.
- Click must open full-screen zoom through the existing `figure[data-zoomable]` lightbox.
- Click again, outside the figure, or Escape must restore normal view.
- Prefer local assets under `site/assets/images/` for stable course material.

## Content Locking Workflow

- Preserve finalized text, image references, and links unless the user explicitly asks to revise them.
- After a module is finalized, run:

```bash
python3 scripts/non_regression_guard.py lock site/chapters/chapter-XX.html --id MXX
```

- After subsequent edits, run:

```bash
python3 scripts/non_regression_guard.py check
```

## Delivery Constraints

- Apply updates incrementally, module by module.
- Keep approved modules stable while working on later modules.
- Reuse the same authoring approach as sibling course `../agents-course`.
