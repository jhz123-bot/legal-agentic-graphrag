import re
from collections import defaultdict
from typing import Dict, List

from src.common.models import GraphEdge, GraphNode
from src.graph.store import InMemoryGraphStore

CHINESE_ALIAS_MAP = {
    "盗窃行为": "盗窃罪",
    "盗窃": "盗窃罪",
    "抢劫": "抢劫罪",
    "抢劫行为": "抢劫罪",
    "诈骗": "诈骗罪",
    "诈骗行为": "诈骗罪",
    "职务侵占罪": "职务侵占",
    "职务侵占行为": "职务侵占",
    "刑法264条": "刑法第二百六十四条",
    "第二百六十四条": "刑法第二百六十四条",
    "刑法263条": "刑法第二百六十三条",
    "第二百六十三条": "刑法第二百六十三条",
    "刑法266条": "刑法第二百六十六条",
    "第二百六十六条": "刑法第二百六十六条",
    "刑法272条": "刑法第二百七十二条",
    "第二百七十二条": "刑法第二百七十二条",
    "民法典577条": "民法典第五百七十七条",
    "第五百七十七条": "民法典第五百七十七条",
    "预期不履行": "预期违约",
    "违约赔偿": "损失赔偿",
    "股东出资": "股东出资义务",
    "董事勤勉": "董事勤勉义务",
    "暴力方式夺取财物": "抢劫罪",
    "持械抢夺财物": "抢劫罪",
    "虚构投资项目骗取钱款": "诈骗罪",
    "骗取钱款": "诈骗罪",
    "利用职务便利侵占单位财物": "职务侵占",
    "职务便利侵占": "职务侵占",
    "未按期缴纳出资": "股东出资义务",
    "按期缴纳出资": "股东出资义务",
    "迟延履行": "违约责任",
    "明确表示不履行主要债务": "预期违约",
    "明确表示不履行": "预期违约",
    "违约损失赔偿范围": "损失赔偿",
    "违约金明显过高": "违约金",
    "催告后承租人仍不付租金": "租赁合同",
    "连续三个月不付租金": "租赁合同",
    "董事违反勤勉义务": "董事勤勉义务",
    "股东未按期缴纳出资": "股东出资义务",
}


def _normalize_alias(name: str) -> str:
    s = name.strip()
    if s in CHINESE_ALIAS_MAP:
        return CHINESE_ALIAS_MAP[s]
    if re.fullmatch(r"第[一二三四五六七八九十百千万零两〇0-9]+条", s):
        if "二百六十四" in s:
            return f"刑法{s}"
        if "五百七十七" in s:
            return f"民法典{s}"
    if s in {"刑法第264条", "刑法264条"}:
        return "刑法第二百六十四条"
    if s in {"刑法第263条", "刑法263条"}:
        return "刑法第二百六十三条"
    if s in {"刑法第266条", "刑法266条"}:
        return "刑法第二百六十六条"
    if s in {"刑法第272条", "刑法272条"}:
        return "刑法第二百七十二条"
    if s in {"民法典第577条", "民法典577条"}:
        return "民法典第五百七十七条"
    return s


def _canonicalize(name: str) -> str:
    normalized = _normalize_alias(name)
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", normalized.lower()).strip()
    tokens = [t for t in cleaned.split() if t not in {"the", "inc", "corp", "corporation", "llc"}]
    return " ".join(tokens) if tokens else cleaned


class EntityLinker:
    def link(self, graph_store: InMemoryGraphStore) -> InMemoryGraphStore:
        buckets: Dict[str, List[GraphNode]] = defaultdict(list)
        for node in graph_store.nodes.values():
            canonical_name = _canonicalize(node.name)
            key = f"{node.entity_type}:{canonical_name}"
            buckets[key].append(node)

        new_nodes: Dict[str, GraphNode] = {}
        id_map: Dict[str, str] = {}

        for group in buckets.values():
            canonical = group[0]
            canonical_name = _normalize_alias(canonical.name)
            merged = GraphNode(
                node_id=canonical.node_id,
                name=canonical_name,
                entity_type=canonical.entity_type,
                aliases=[],
                mentions=[],
                metadata=dict(canonical.metadata),
            )
            for node in group:
                id_map[node.node_id] = canonical.node_id
                alias_name = _normalize_alias(node.name)
                if alias_name != canonical_name and alias_name not in merged.aliases:
                    merged.aliases.append(alias_name)
                for alias in node.aliases:
                    norm_alias = _normalize_alias(alias)
                    if norm_alias != canonical_name and norm_alias not in merged.aliases:
                        merged.aliases.append(norm_alias)
                merged.mentions.extend(node.mentions)
                merged.metadata.update(node.metadata)
            new_nodes[canonical.node_id] = merged

        new_edges: List[GraphEdge] = []
        seen = set()
        for edge in graph_store.edges:
            src = id_map.get(edge.source, edge.source)
            tgt = id_map.get(edge.target, edge.target)
            key = (src, tgt, edge.relation, edge.evidence)
            if key in seen:
                continue
            seen.add(key)
            new_edges.append(
                GraphEdge(
                    source=src,
                    target=tgt,
                    relation=edge.relation,
                    weight=edge.weight,
                    evidence=edge.evidence,
                )
            )

        graph_store.nodes = new_nodes
        graph_store.edges = []
        graph_store.adjacency.clear()
        for edge in new_edges:
            graph_store.add_edge(edge)
        return graph_store
