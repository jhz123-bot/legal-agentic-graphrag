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
    normalize_articles_in_text,
    normalize_entity_alias,
    normalize_keywords,
)


def clean_law_item(item: dict) -> dict:
    law_name = clean_text(str(item.get("law_name", "")))
    article_no = normalize_article_name(clean_text(str(item.get("article_no", ""))), default_law_name=law_name)
    text = clean_text(str(item.get("text", "")))
    text = normalize_entity_alias(text)
    text = normalize_articles_in_text(text, default_law_name=law_name)

    return {
        "doc_id": clean_text(str(item.get("doc_id", ""))),
        "law_name": law_name,
        "article_no": article_no,
        "chapter": clean_text(str(item.get("chapter", ""))),
        "effective_date": clean_text(str(item.get("effective_date", ""))),
        "text": text,
        "keywords": normalize_keywords(item.get("keywords", []), default_law_name=law_name),
    }


def clean_file(path: Path, out_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        cleaned = [clean_law_item(it) for it in data if isinstance(it, dict)]
    elif isinstance(data, dict):
        cleaned = clean_law_item(data)
    else:
        cleaned = data

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize law layer JSON files")
    parser.add_argument("--input-dir", default="data/legal_kb/laws")
    parser.add_argument("--output-dir", default="data/legal_kb_cleaned/laws")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    for path in sorted(in_dir.glob("*.json")):
        clean_file(path, out_dir)
    print(f"clean_laws done: {in_dir} -> {out_dir}")


if __name__ == "__main__":
    main()

