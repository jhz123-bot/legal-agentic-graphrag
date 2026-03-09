import json
from pathlib import Path

root = Path('data/raw')
for d in ['statutes', 'cases', 'faqs']:
    (root / d).mkdir(parents=True, exist_ok=True)

law_pool = [
    ('中华人民共和国刑法', '分则', '盗窃罪'),
    ('中华人民共和国刑法', '分则', '诈骗罪'),
    ('中华人民共和国民法典', '合同编', '违约责任'),
    ('中华人民共和国民法典', '合同编', '租赁合同'),
    ('中华人民共和国民法典', '侵权责任编', '侵权责任'),
    ('中华人民共和国公司法', '公司设立和组织机构', '公司治理'),
    ('中华人民共和国劳动合同法', '劳动合同的履行和变更', '劳动争议'),
    ('最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释', '总则', '买卖合同'),
    ('最高人民法院关于审理民间借贷案件适用法律若干问题的规定', '总则', '借贷纠纷'),
]


def article_no(idx: int) -> str:
    return f'第{300 + idx}条'


statutes = []
for i in range(1, 121):
    law_name, chapter, topic = law_pool[(i - 1) % len(law_pool)]
    art = article_no(i)
    title = f'{law_name}{art}'
    content = (
        f'{title}\n'
        f'第{i}页 共120页\n'
        f'{law_name}\n'
        f'{chapter}\n'
        f'{art}：针对{topic}争议，行为人应依法承担相应责任。\n'
        f'人民法院审理时，应综合行为性质、损失后果、过错程度与证据情况作出认定。\n'
        f'本条适用于与{topic}相关的法律关系处理。\n'
        f'页码:{i}\n'
    )
    statutes.append(
        {
            'doc_id': f'statute_{i:03d}',
            'title': title,
            'law_name': law_name,
            'chapter': chapter,
            'article_no': art,
            'source_type': 'statute',
            'content': content,
        }
    )

case_types = ['盗窃案件', '合同违约案件', '租赁纠纷案件', '侵权责任案件', '公司纠纷案件', '劳动争议案件', '买卖合同案件', '民间借贷案件']
courts = ['某市中级人民法院', '某区人民法院', '某省高级人民法院', '某铁路运输法院']

cases = []
for i in range(1, 101):
    ctype = case_types[(i - 1) % len(case_types)]
    court = courts[(i - 1) % len(courts)]
    title = f'{ctype}摘要第{i}号'
    dispute_focus = '责任承担范围、证据证明力及法律适用顺序'
    judgment = '支持有证据支持的诉请，对证据不足部分不予支持'
    content = (
        f'{title}\n'
        '中国裁判文书网\n'
        f'第{i}页\n'
        f'案情：张某与甲公司围绕{ctype}发生争议，李某、乙公司参与交易或履行。\n'
        f'争议焦点：{dispute_focus}。\n'
        f'法院认为：{court}认为应结合合同履行情况、过错程度、损失结果与举证责任分配综合判断。\n'
        f'裁判结果：{judgment}。\n'
        f'页码:{i}\n'
    )
    cases.append(
        {
            'doc_id': f'case_{i:03d}',
            'case_id': f'（2025）示例案字第{i:04d}号',
            'title': title,
            'court': court,
            'case_type': ctype,
            'dispute_focus': dispute_focus,
            'judgment': judgment,
            'source_type': 'case',
            'content': content,
        }
    )

faq_topics = ['盗窃罪', '诈骗罪', '合同违约', '租赁纠纷', '侵权责任', '公司纠纷', '劳动争议', '买卖合同', '民间借贷']
faqs = []
for i in range(1, 41):
    topic = faq_topics[(i - 1) % len(faq_topics)]
    q = f'{topic}中，当事人应如何依法主张权利？第{i}问'
    a = '一般应先固定证据，再依据相关法律条文确定请求权基础；协商不成时，可依法诉讼或仲裁。'
    faqs.append(
        {
            'faq_id': f'faq_{i:03d}',
            'question': q,
            'answer': a,
            'topic': topic,
            'source_type': 'faq',
        }
    )

with (root / 'statutes' / 'statutes_raw_round2.jsonl').open('w', encoding='utf-8') as f:
    for row in statutes:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

with (root / 'cases' / 'cases_raw_round2.jsonl').open('w', encoding='utf-8') as f:
    for row in cases:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

with (root / 'faqs' / 'faqs_raw_round2.jsonl').open('w', encoding='utf-8') as f:
    for row in faqs:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print({'statutes': len(statutes), 'cases': len(cases), 'faqs': len(faqs)})
