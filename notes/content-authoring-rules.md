# CUDA Course Content Authoring Rules

Confirmed workflow:

1. Build the course one module at a time.
2. Use the official syllabus as the structural contract for each module.
3. Use the main book as the primary source.
4. Supplement with NVIDIA, CUDA Best Practices, PyTorch, and curated web resources.
5. Keep the existing site structure aligned with the syllabus.
6. Use the same approach as sibling repository `../agents-course`.
7. Use exactly two content levels: numbered `h2` sections and numbered `h3` subsections.
8. Render each subsection as a slide-like rectangular block.
9. Keep the left outline sidebar navigable and collapsible.
10. Make all figures numbered and clickable.
11. Use full-screen figure zoom through `figure[data-zoomable]`.
12. Lock finalized module content with `scripts/non_regression_guard.py`.
13. Run non-regression checks after changes to protect finalized modules.

Current syllabus note:

- The user referred to `resources/sillabo.txt`.
- The file currently present in the workspace is `resources/sillabo-corso.txt`.
- Until `resources/sillabo.txt` exists, use `resources/sillabo-corso.txt` and mention the mismatch when relevant.
