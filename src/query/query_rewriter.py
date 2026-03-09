from __future__ import annotations

import re
from typing import List


_EXPANSION_MAP = {
    "合同违约": ["违约责任", "损害赔偿", "继续履行", "解除合同"],
    "盗窃": ["盗窃罪", "刑法", "量刑", "数额"],
    "诈骗": ["诈骗罪", "刑法", "构成要件"],
    "租赁": ["租赁合同", "租金", "承租人", "出租人"],
    "侵权": ["侵权责任", "过错", "赔偿"],
}


def rewrite_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q

    extras: List[str] = []
    for key, vals in _EXPANSION_MAP.items():
        if key in q:
            extras.extend(vals)

    # Clarify short queries with legal retrieval intent hints.
    if len(q) <= 10 and not re.search(r"法条|条文|责任|构成|要件", q):
        extras.extend(["法律条款", "裁判要点"])

    extras = list(dict.fromkeys(extras))
    if not extras:
        return q
    return f"{q} {' '.join(extras)}"
