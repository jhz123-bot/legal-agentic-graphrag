import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.dedup import dedup_cases, dedup_faqs, dedup_statutes
from src.ingestion.legal_structurer import structure_case, structure_faq, structure_statute
from src.ingestion.legal_text_cleaner import clean_legal_text


def _iter_json_records(path: Path) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        return

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        yield payload


def _load_raw_items(raw_dir: Path, phase: str) -> List[Dict]:
    items: List[Dict] = []
    if not raw_dir.exists():
        return items
    preferred = sorted(raw_dir.glob(f"*_{phase}.json*"))
    targets = preferred if preferred else sorted(raw_dir.glob("*.json*"))
    for path in targets:
        items.extend(list(_iter_json_records(path)))
    return items


def _clean_statute(item: Dict, phase: str) -> Dict:
    doc_id = str(item.get("doc_id", "")).strip()
    title = str(item.get("title", "")).strip()
    content = clean_legal_text(str(item.get("content", "")), title=title)
    base = {
        "doc_id": doc_id,
        "title": title,
        "source_type": "statute",
        "content": content,
    }
    if phase in {"round2", "round3", "round4"}:
        base.update(structure_statute(item, content))
    return base


def _clean_case(item: Dict, phase: str) -> Dict:
    doc_id = str(item.get("doc_id", "")).strip()
    title = str(item.get("title", "")).strip()
    content = clean_legal_text(str(item.get("content", "")), title=title)
    base = {
        "doc_id": doc_id,
        "title": title,
        "source_type": "case",
        "content": content,
    }
    if phase in {"round2", "round3", "round4"}:
        base.update(structure_case(item, content))
    return base


def _clean_faq(item: Dict, phase: str) -> Dict:
    faq_id = str(item.get("faq_id", "")).strip()
    question = clean_legal_text(str(item.get("question", "")))
    answer = clean_legal_text(str(item.get("answer", "")))
    base = {
        "faq_id": faq_id,
        "question": question,
        "answer": answer,
        "source_type": "faq",
    }
    if phase in {"round2", "round3", "round4"}:
        base.update(structure_faq(item))
    return base


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for row in rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")


def _missing_field_stats(rows: List[Dict], required_fields: List[str]) -> Dict[str, int]:
    stats = {f: 0 for f in required_fields}
    for row in rows:
        for f in required_fields:
            if not str(row.get(f, "")).strip():
                stats[f] += 1
    return stats


def _length_stats(rows: List[Dict], key: str, short_threshold: int, long_threshold: int) -> Dict[str, int]:
    short_count = 0
    long_count = 0
    for row in rows:
        n = len(str(row.get(key, "")))
        if n < short_threshold:
            short_count += 1
        if n > long_threshold:
            long_count += 1
    return {"too_short": short_count, "too_long": long_count}


def _apply_dedup(phase: str, statutes: List[Dict], cases: List[Dict], faqs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, object]]:
    dedup_stats: Dict[str, object] = {
        "statutes": {"removed": 0, "reasons": {}},
        "cases": {"removed": 0, "reasons": {}},
        "faqs": {"removed": 0, "reasons": {}},
    }
    if phase not in {"round3", "round4"}:
        return statutes, cases, faqs, dedup_stats

    statute_res = dedup_statutes(statutes)
    case_res = dedup_cases(cases)
    faq_res = dedup_faqs(faqs)

    dedup_stats["statutes"] = {"removed": statute_res.removed, "reasons": statute_res.reasons}
    dedup_stats["cases"] = {"removed": case_res.removed, "reasons": case_res.reasons}
    dedup_stats["faqs"] = {"removed": faq_res.removed, "reasons": faq_res.reasons}
    return statute_res.kept, case_res.kept, faq_res.kept, dedup_stats


def prepare_corpus(
    phase: str,
    raw_root: Path,
    cleaned_round_dir: Path,
    processed_dir: Path,
    metadata_dir: Path,
) -> Dict[str, object]:
    statutes_raw = _load_raw_items(raw_root / "statutes", phase=phase)
    cases_raw = _load_raw_items(raw_root / "cases", phase=phase)
    faqs_raw = _load_raw_items(raw_root / "faqs", phase=phase)

    statutes_clean = [_clean_statute(x, phase=phase) for x in statutes_raw]
    cases_clean = [_clean_case(x, phase=phase) for x in cases_raw]
    faqs_clean = [_clean_faq(x, phase=phase) for x in faqs_raw]

    statutes_clean, cases_clean, faqs_clean, dedup_stats = _apply_dedup(
        phase=phase,
        statutes=statutes_clean,
        cases=cases_clean,
        faqs=faqs_clean,
    )

    _write_jsonl(cleaned_round_dir / "statutes" / f"statutes_{phase}.jsonl", statutes_clean)
    _write_jsonl(cleaned_round_dir / "cases" / f"cases_{phase}.jsonl", cases_clean)
    _write_jsonl(cleaned_round_dir / "faqs" / f"faqs_{phase}.jsonl", faqs_clean)

    _write_jsonl(processed_dir / f"{phase}_statutes.jsonl", statutes_clean)
    _write_jsonl(processed_dir / f"{phase}_cases.jsonl", cases_clean)
    _write_jsonl(processed_dir / f"{phase}_faqs.jsonl", faqs_clean)

    added_fields = {
        "statute": ["law_name", "chapter", "article_no"] if phase in {"round2", "round3", "round4"} else [],
        "case": ["case_id", "court", "case_type", "dispute_focus", "judgment"] if phase in {"round2", "round3", "round4"} else [],
        "faq": ["topic"] if phase in {"round2", "round3", "round4"} else [],
    }

    summary: Dict[str, object] = {
        "phase": phase,
        "counts": {
            "statutes": len(statutes_clean),
            "cases": len(cases_clean),
            "faqs": len(faqs_clean),
        },
        "raw_counts": {
            "statutes": len(statutes_raw),
            "cases": len(cases_raw),
            "faqs": len(faqs_raw),
        },
        "raw_root": str(raw_root),
        "cleaned_round_dir": str(cleaned_round_dir),
        "processed_dir": str(processed_dir),
        "added_fields": added_fields,
        "dedup": dedup_stats,
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"{phase}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if phase in {"round3", "round4"}:
        statute_article_conflicts = len(statutes_clean) - len({str(x.get("article_no", "")).strip() for x in statutes_clean if str(x.get("article_no", "")).strip()})
        case_id_conflicts = len(cases_clean) - len({str(x.get("case_id", "")).strip() for x in cases_clean if str(x.get("case_id", "")).strip()})
        faq_question_conflicts = len(faqs_clean) - len({str(x.get("question", "")).strip() for x in faqs_clean if str(x.get("question", "")).strip()})
        quality_report = {
            "phase": phase,
            "missing_fields": {
                "statutes": _missing_field_stats(statutes_clean, ["doc_id", "title", "law_name", "article_no", "content"]),
                "cases": _missing_field_stats(cases_clean, ["doc_id", "case_id", "title", "court", "case_type", "content"]),
                "faqs": _missing_field_stats(faqs_clean, ["faq_id", "question", "answer", "topic"]),
            },
            "length_checks": {
                "statutes": _length_stats(statutes_clean, key="content", short_threshold=40, long_threshold=1200),
                "cases": _length_stats(cases_clean, key="content", short_threshold=60, long_threshold=2000),
                "faqs_answer": _length_stats(faqs_clean, key="answer", short_threshold=15, long_threshold=500),
            },
            "conflict_checks": {
                "statute_article_no_conflicts": statute_article_conflicts,
                "case_id_conflicts": case_id_conflicts,
                "faq_question_conflicts": faq_question_conflicts,
            },
            "dedup": dedup_stats,
        }
        (metadata_dir / f"{phase}_quality_report.json").write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal corpus preparation by phase")
    parser.add_argument("--phase", default="round1", choices=["round1", "round2", "round3", "round4"], help="pipeline phase")
    parser.add_argument("--raw-root", default="data/raw", help="raw data root")
    parser.add_argument("--cleaned-root", default="data/cleaned", help="cleaned output root")
    parser.add_argument("--processed-root", default="data/processed", help="processed output root")
    parser.add_argument("--metadata-root", default="data/metadata", help="metadata output root")
    args = parser.parse_args()

    summary = prepare_corpus(
        phase=args.phase,
        raw_root=Path(args.raw_root),
        cleaned_round_dir=Path(args.cleaned_root) / args.phase,
        processed_dir=Path(args.processed_root),
        metadata_dir=Path(args.metadata_root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
