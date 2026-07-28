#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from html import unescape
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_DIR = ROOT / "tests" / "non-regression" / "locks"
MODULE_FILE_RE = re.compile(r"^site/chapters/chapter-\d+\.html$")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

REQUIRED_UI_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("Left outline nav", re.compile(r'id="outline-nav"', re.IGNORECASE)),
    ("Print button", re.compile(r'data-print', re.IGNORECASE)),
]


def normalize(text: str) -> str:
    text = unescape(text)
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def extract_heading_and_core_texts(html: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"<h1[^>]*>(.*?)</h1>",
        r"<h2[^>]*>(.*?)</h2>",
        r"<h3[^>]*>(.*?)</h3>",
        r"<p[^>]*class=[\"'][^\"']*hero-copy[^\"']*[\"'][^>]*>(.*?)</p>",
    ]
    for pattern in patterns:
        for raw in re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            text = normalize(raw)
            if text:
                candidates.append(text)

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def extract_images(html: str) -> list[str]:
    srcs = re.findall(r"<img[^>]+src=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for src in srcs:
        src = src.strip()
        if not src or src in seen:
            continue
        seen.add(src)
        out.append(src)
    return out


def extract_links(html: str) -> list[str]:
    hrefs = re.findall(r"<a[^>]+href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        href = href.strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out


def lock_id_from_target(target: Path) -> str:
    match = re.search(r"chapter-(\d+)", target.name)
    if match:
        return f"M{int(match.group(1))}"
    return target.stem


def missing_items(required: Iterable[str], haystack: str) -> list[str]:
    norm_hay = normalize(haystack).lower()
    missing: list[str] = []
    for item in required:
        if normalize(item).lower() not in norm_hay:
            missing.append(item)
    return missing


def extract_heading_tags(html: str, tag: str) -> list[str]:
    return re.findall(fr"<{tag}\b[^>]*>.*?</{tag}>", html, flags=re.IGNORECASE | re.DOTALL)


def check_required_ui_markers(html: str) -> list[str]:
    issues: list[str] = []
    for label, pattern in REQUIRED_UI_MARKERS:
        if not pattern.search(html):
            issues.append(label)
    return issues


def check_two_level_structure(html: str) -> list[str]:
    issues: list[str] = []
    h2s = extract_heading_tags(html, "h2")
    h3s = extract_heading_tags(html, "h3")
    if not h2s:
        issues.append("Missing h2 headings")
    if not h3s:
        issues.append("Missing h3 headings")
    if re.search(r"<h[4-6]\b", html, flags=re.IGNORECASE):
        issues.append("Found heading levels deeper than h3")
    first_h2 = re.search(r"<h2\b", html, flags=re.IGNORECASE)
    first_h3 = re.search(r"<h3\b", html, flags=re.IGNORECASE)
    if first_h3 and not first_h2:
        issues.append("Found h3 headings but no h2 headings")
    elif first_h2 and first_h3 and first_h3.start() < first_h2.start():
        issues.append("Found h3 before first h2")
    return issues


def check_heading_numbering(html: str) -> list[str]:
    issues: list[str] = []
    heading_matches = re.finditer(
        r"<(h2|h3)\b[^>]*>(.*?)</\1>", html, flags=re.IGNORECASE | re.DOTALL
    )
    numbered: list[tuple[str, tuple[int, ...], str]] = []
    num_re = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s+")
    for match in heading_matches:
        tag = match.group(1).lower()
        text = normalize(match.group(2))
        number = num_re.match(text)
        if not number:
          continue
        parts = tuple(int(p) for p in number.group(1).split("."))
        numbered.append((tag, parts, text))

    seen_ids: set[tuple[str, tuple[int, ...]]] = set()
    for tag, parts, text in numbered:
        key = (tag, parts)
        if key in seen_ids:
            issues.append(f"Duplicate numbering in {tag}: {text}")
            continue
        seen_ids.add(key)

    last_seen: dict[tuple[str, tuple[int, ...]], int] = {}
    for tag, parts, text in numbered:
        prefix = parts[:-1]
        key = (tag, prefix)
        current = parts[-1]
        if key in last_seen and current != last_seen[key] + 1:
            issues.append(f"Non-sequential numbering in {tag}: {text}")
        last_seen[key] = current

    h2_numbers = {parts for tag, parts, _ in numbered if tag == "h2"}
    for tag, parts, text in numbered:
        if tag != "h3":
            continue
        if len(parts) < 2:
            issues.append(f"h3 heading should use hierarchical numbering: {text}")
            continue
        if parts[:-1] not in h2_numbers:
            issues.append(f"h3 heading parent not found in h2 numbering: {text}")
    return issues


def check_module_flow(html: str) -> list[str]:
    issues: list[str] = []
    headings = [normalize(h) for h in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", html, flags=re.IGNORECASE | re.DOTALL)]
    if not any("Learning outcomes" == h for h in headings):
        issues.append("Missing 'Learning outcomes' section")
    if not any("Scope" == h for h in headings):
        issues.append("Missing 'Scope' section")
    if not any(h.startswith("1. ") for h in headings):
        issues.append("Missing numbered module content section")

    info_grid_pos = html.find('class="panel info-grid"')
    article_pos = html.find("<article")
    if info_grid_pos == -1:
        issues.append("Missing info-grid panel")
    elif article_pos != -1 and info_grid_pos < article_pos:
        issues.append("info-grid panel must be inside article")
    return issues


def check_ui_logic(root_path: Path) -> list[str]:
    issues: list[str] = []
    js_file = root_path / "site" / "assets" / "js" / "site.js"
    css_file = root_path / "site" / "assets" / "css" / "site.css"
    if not js_file.exists():
        issues.append("Missing site/assets/js/site.js")
    else:
        js = js_file.read_text(encoding="utf-8")
        for needle, label in [
            ("outline-toggle-spacer", "JS outline spacer logic"),
            ("toggle.addEventListener", "JS outline toggle listener"),
            ("outline-collapse", "JS outline collapse button"),
            ("[data-print]", "JS print binding"),
        ]:
            if needle not in js:
                issues.append(f"Missing {label}")

    if not css_file.exists():
        issues.append("Missing site/assets/css/site.css")
    else:
        css = css_file.read_text(encoding="utf-8")
        for needle, label in [
            (".outline-toggle-spacer", "CSS outline spacer"),
            (".print-fixed", "CSS fixed print button"),
            (".outline-panel[data-collapsed=\"true\"] #outline-nav", "CSS collapsed outline state"),
        ]:
            if needle not in css:
                issues.append(f"Missing {label}")
    return issues


def check_image_links(html: str, module_file: Path) -> list[str]:
    issues: list[str] = []
    for src in extract_images(html):
        if src.startswith("http"):
            continue
        img_path = (module_file.parent / src).resolve()
        if not img_path.exists():
            issues.append(f"Broken image link: {src}")
    return issues


def check_local_links(html: str, module_file: Path) -> list[str]:
    issues: list[str] = []
    for href in extract_links(html):
        if re.match(r"^[a-z]+://", href, flags=re.IGNORECASE):
            continue
        if href.startswith("//"):
            continue
        target = (module_file.parent / href).resolve()
        if not target.exists():
            issues.append(f"Broken local link: {href}")
    return issues


def cmd_lock(args: argparse.Namespace) -> int:
    target = (ROOT / args.target).resolve() if not Path(args.target).is_absolute() else Path(args.target)
    if not target.exists():
        print(f"[error] Target file not found: {target}")
        return 1

    html = target.read_text(encoding="utf-8")
    texts = extract_heading_and_core_texts(html)
    images = extract_images(html)
    links = extract_links(html)
    if args.max_texts > 0:
        texts = texts[:args.max_texts]

    lock_id = args.id or lock_id_from_target(target)
    lock_dir = Path(args.lock_dir)
    if not lock_dir.is_absolute():
        lock_dir = (ROOT / lock_dir).resolve()
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_path = lock_dir / f"{lock_id}.lock.json"
    payload = {
        "id": lock_id,
        "file": str(target.relative_to(ROOT)),
        "required_texts": texts,
        "required_images": images,
        "required_links": links,
    }
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] Lock written: {lock_path}")
    print(f"     required_texts: {len(texts)}")
    print(f"     required_images: {len(images)}")
    print(f"     required_links: {len(links)}")
    return 0


def load_lock_files(lock_dir: Path, only_id: str | None) -> list[Path]:
    if only_id:
        p = lock_dir / f"{only_id}.lock.json"
        return [p] if p.exists() else []
    return sorted(lock_dir.glob("*.lock.json"))


def cmd_check(args: argparse.Namespace) -> int:
    lock_dir = Path(args.lock_dir)
    if not lock_dir.is_absolute():
        lock_dir = (ROOT / lock_dir).resolve()

    failures = 0
    lock_files = load_lock_files(lock_dir, args.id)
    if not lock_files:
        print(f"[warn] No lock files found in {lock_dir}")
    else:
        for lock_file in lock_files:
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            rel = data.get("file", "")
            target = (ROOT / rel).resolve()
            print(f"\n[check] {data.get('id', lock_file.stem)} -> {rel}")
            if not target.exists():
                print(f"  [fail] Target file missing: {target}")
                failures += 1
                continue

            html = target.read_text(encoding="utf-8")
            miss_texts = missing_items(data.get("required_texts", []), html)
            current_imgs = set(extract_images(html))
            miss_imgs = [img for img in data.get("required_images", []) if img not in current_imgs]
            current_links = set(extract_links(html))
            miss_links = [link for link in data.get("required_links", []) if link not in current_links]

            if miss_texts or miss_imgs or miss_links:
                failures += 1
                if miss_texts:
                    print(f"  [fail] Missing required texts ({len(miss_texts)}):")
                    for item in miss_texts:
                        print(f"    - {item}")
                if miss_imgs:
                    print(f"  [fail] Missing required images ({len(miss_imgs)}):")
                    for item in miss_imgs:
                        print(f"    - {item}")
                if miss_links:
                    print(f"  [fail] Missing required links ({len(miss_links)}):")
                    for item in miss_links:
                        print(f"    - {item}")
            else:
                print("  [ok] No regressions detected for locked texts/images/links")

    print("\n[logic] Checking shared UI logic...")
    logic_issues = check_ui_logic(ROOT)
    if logic_issues:
        failures += 1
        for issue in logic_issues:
            print(f"  [fail] {issue}")
    else:
        print("  [ok] Shared JS/CSS logic is present")

    module_files = sorted((ROOT / "site" / "chapters").glob("chapter-*.html"))
    if not module_files:
        print("\n[warn] No module pages found under site/chapters")
    else:
        for module_file in module_files:
            rel = str(module_file.relative_to(ROOT))
            if not MODULE_FILE_RE.match(rel):
                continue
            html = module_file.read_text(encoding="utf-8")
            issues = (
                check_required_ui_markers(html)
                + check_two_level_structure(html)
                + check_heading_numbering(html)
                + check_module_flow(html)
                + check_image_links(html, module_file)
                + check_local_links(html, module_file)
            )
            print(f"\n[structure] {rel}")
            if issues:
                failures += 1
                for issue in issues:
                    print(f"  [fail] {issue}")
            else:
                print("  [ok] UI, structure, numbering, and local links are valid")

    if failures:
        print(f"\n[result] FAIL - {failures} check group(s) with regressions")
        return 1
    print("\n[result] PASS - all modules and shared logic are compliant")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lock/check non-regression for finalized CUDA course modules")
    sub = parser.add_subparsers(dest="command", required=True)

    p_lock = sub.add_parser("lock", help="Create/update a lock file for a module page")
    p_lock.add_argument("target", help="Target HTML file, e.g. site/chapters/chapter-01.html")
    p_lock.add_argument("--id", help="Lock id, e.g. M1")
    p_lock.add_argument("--max-texts", type=int, default=24, help="Max required texts to persist")
    p_lock.add_argument("--lock-dir", default=str(DEFAULT_LOCK_DIR.relative_to(ROOT)), help="Lock file directory")
    p_lock.set_defaults(func=cmd_lock)

    p_check = sub.add_parser("check", help="Check all lock files for regressions")
    p_check.add_argument("--id", help="Check a single lock id")
    p_check.add_argument("--lock-dir", default=str(DEFAULT_LOCK_DIR.relative_to(ROOT)), help="Lock file directory")
    p_check.set_defaults(func=cmd_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
