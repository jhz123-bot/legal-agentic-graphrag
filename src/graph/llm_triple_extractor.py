import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt

ALLOWED_ENTITY_TYPES = {"Crime", "LawArticle", "Case", "Court", "Circumstance", "Liability", "Party"}
ALLOWED_RELATION_TYPES = {"APPLIES_TO", "CITES", "HAS_CIRCUMSTANCE", "DECIDED_BY", "INVOLVES", "BEARS_LIABILITY"}


class LLMTripleExtractor:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 8) -> None:
        provider_name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider_name == "dashscope":
            provider_name = "bailian"
        if provider_name == "bailian":
            default_model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        else:
            default_model = os.getenv("OLLAMA_TRIPLE_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
        self.provider_name = provider_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or default_model
        self.timeout = timeout
        self.provider = get_llm_provider(provider=self.provider_name, timeout=timeout)
        self.cache_dir = Path("outputs/cache/triples")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, text: str) -> str:
        return render_prompt("triple_extraction", text=text)

    def _cache_key(self, text: str) -> str:
        raw = f"{self.provider_name}\n{self.model}\n{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text)}.json"

    def _load_cache(self, text: str) -> Dict[str, List[Dict[str, str]]] | None:
        path = self._cache_path(text)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "entities" in data and "triples" in data:
                return data
        except Exception:
            return None
        return None

    def _save_cache(self, text: str, data: Dict[str, List[Dict[str, str]]]) -> None:
        path = self._cache_path(text)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        if not raw:
            return {"entities": [], "triples": []}
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            candidate = candidate.removesuffix("```").strip()
        try:
            return json.loads(candidate)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", candidate)
            if not m:
                return {"entities": [], "triples": []}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"entities": [], "triples": []}

    def _normalize_entity_name(self, name: str, typ: str) -> str:
        s = name.strip().replace("《", "").replace("》", "")
        s = s.replace("中华人民共和国", "")

        if typ == "Crime":
            if "盗窃" in s:
                return "盗窃罪"
            if "诈骗" in s:
                return "诈骗罪"
        if typ == "Liability":
            if "违约" in s:
                return "违约责任"
            if "侵权" in s:
                return "侵权责任"
        if typ == "LawArticle":
            if re.fullmatch(r"第[一二三四五六七八九十百千万零两〇0-9]+条", s):
                if "二百六十四" in s:
                    return f"刑法{s}"
                if "五百七十七" in s:
                    return f"民法典{s}"
            if "刑法" in s and "二百六十四" in s:
                return "刑法第二百六十四条"
            if "民法典" in s and "五百七十七" in s:
                return "民法典第五百七十七条"
        return s

    def _validate(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        entities_raw = data.get("entities", []) if isinstance(data, dict) else []
        triples_raw = data.get("triples", []) if isinstance(data, dict) else []

        entities: List[Dict[str, str]] = []
        seen_entities: set[Tuple[str, str]] = set()
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).strip()
            if typ not in ALLOWED_ENTITY_TYPES:
                continue
            name = self._normalize_entity_name(str(item.get("name", "")).strip(), typ)
            if not name:
                continue
            key = (name, typ)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            entities.append({"name": name, "type": typ})

        triples: List[Dict[str, str]] = []
        seen_triples: set[Tuple[str, str, str]] = set()
        for item in triples_raw:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            rel = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subj or not obj or rel not in ALLOWED_RELATION_TYPES:
                continue
            key = (subj, rel, obj)
            if key in seen_triples:
                continue
            seen_triples.add(key)
            triples.append({"subject": subj, "relation": rel, "object": obj})

        return {"entities": entities, "triples": triples}

    def extract_triples(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        if not text.strip():
            return {"entities": [], "triples": []}

        cached = self._load_cache(text)
        if cached is not None:
            return cached

        try:
            raw = self.provider.generate(prompt=self._build_prompt(text), model=self.model, temperature=0.0)
            parsed = self._extract_json(raw)
            validated = self._validate(parsed)
            self._save_cache(text, validated)
            return validated
        except Exception:
            empty = {"entities": [], "triples": []}
            self._save_cache(text, empty)
            return empty
