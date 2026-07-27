# Non-regression Locks (CUDA Course Modules)

When a module is finalized, create a lock that freezes:
- required key texts
- required image references (`<img src=...>`)
- required links already present in the module

Regression checks also enforce baseline module UI and structure on every
`site/chapters/chapter-*.html` page:
- left outline navigation is present
- print controls are present
- content keeps a two-level heading hierarchy (`h2` + `h3`, no deeper levels)
- heading numbering is validated
- expected module sections are present (`Learning outcomes`, `Scope`, `Syllabus Topics`, `Guided Exercises`)
- local links and image paths resolve correctly

## Create or update a lock

```bash
cd /home/thimoty/git/cuda-course
python3 scripts/non_regression_guard.py lock site/chapters/chapter-01.html --id M1
```

## Run checks

```bash
cd /home/thimoty/git/cuda-course
python3 scripts/non_regression_guard.py check
```

Check one module only:

```bash
python3 scripts/non_regression_guard.py check --id M1
```

## Workflow

1. Complete module content.
2. Confirm module is final.
3. Run `lock` for that module.
4. Run `check` after every subsequent change.
