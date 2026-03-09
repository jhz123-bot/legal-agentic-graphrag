import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_cleaning.normalize_entities import (
    clean_text,
    normalize_article_name,
    normalize_entity_alias,
    normalize_keywords,
)


def clean_case_item(item: dict) -> dict:
    facts = normalize_entity_alias(clean_text(str(item.get("facts", ""))))
    holding = normalize_entity_alias(clean_text(str(item.get("holding", ""))))
    issues = [normalize_entity_alias(clean_text(str(x))) for x in item.get("issues", [])]
    cited_laws = [normalize_article_name(clean_text(str(x))) for x in item.get("cited_laws", [])]

    return {
        "case_id": clean_text(str(item.get("case_id", ""))),
        "case_type": clean_text(str(item.get("case_type", ""))),
        "court": clean_text(str(item.get("court", ""))),
        "facts": facts,
        "issues": [x for x in issues if x],
        "holding": holding,
        "cited_laws": [x for x in cited_laws if x],
        "keywords": normalize_keywords(item.get("keywords", [])),
    }


def clean_file(path: Path, out_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        cleaned = [clean_case_item(it) for it in data if isinstance(it, dict)]
    elif isinstance(data, dict):
        cleaned = clean_case_item(data)
    else:
        cleaned = data

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize case layer JSON files")
    parser.add_argument("--input-dir", default="data/legal_kb/cases")
    parser.add_argument("--output-dir", default="data/legal_kb_cleaned/cases")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    for path in sorted(in_dir.glob("*.json")):
        clean_file(path, out_dir)
    print(f"clean_cases done: {in_dir} -> {out_dir}")


if __name__ == "__main__":
    main()

