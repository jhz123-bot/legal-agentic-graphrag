import json
from pathlib import Path

root = Path('data/raw')
for d in ['statutes', 'cases', 'faqs']:
    (root / d).mkdir(parents=True, exist_ok=True)

statute_topics = [
    ('中华人民共和国刑法', '盗窃罪', '财产犯罪'),
    ('中华人民共和国刑法', '诈骗罪', '财产犯罪'),
    ('中华人民共和国民法典', '违约责任', '合同编'),
    ('中华人民共和国民法典', '租赁合同', '合同编'),
    ('中华人民共和国民法典', '侵权责任', '侵权责任编'),
    ('中华人民共和国公司法', '公司决议效力', '公司治理'),
    ('中华人民共和国劳动合同法', '劳动争议', '劳动关系'),
]


def article_no(i: int) -> str:
    return f'第{200 + i}条'


statutes = []
for i in range(1, 51):
    law, topic, chapter = statute_topics[(i - 1) % len(statute_topics)]
    art = article_no(i)
    title = f'{law}{art}'
    content = (
        f'{title}\n'
        f'第{i}页 共50页\n'
        f'{law}\n'
        f'【{topic}】{chapter}相关规则。\n'
        f'{art}：行为人实施与{topic}相关的行为，依法承担相应法律责任。\n'
        '在情节、后果、主观状态等因素下，人民法院应结合证据依法认定。\n'
        f'页码:{i}\n'
    )
    statutes.append(
        {
            'doc_id': f'statute_{i:03d}',
            'title': title,
            'source_type': 'statute',
            'content': content,
        }
    )

case_types = ['盗窃案件', '合同违约案件', '租赁纠纷案件', '侵权责任案件', '公司纠纷案件', '劳动争议案件']
courts = ['某市中级人民法院', '某区人民法院', '某省高级人民法院']

cases = []
for i in range(1, 31):
    ctype = case_types[(i - 1) % len(case_types)]
    court = courts[(i - 1) % len(courts)]
    title = f'{ctype}摘要第{i}号'
    content = (
        f'{title}\n'
        '中国裁判文书网\n'
        f'第{i}页\n'
        f'案情：张某与甲公司就{ctype}发生争议，李某参与相关交易。\n'
        '争议焦点：责任承担范围、证据证明力及法律适用顺序。\n'
        f'法院认为：{court}认为应根据合同履行、过错程度与损失结果综合判断。\n'
        '裁判结果：支持合理请求，驳回缺乏证据部分。\n'
        f'页码:{i}\n'
    )
    cases.append(
        {
            'doc_id': f'case_{i:03d}',
            'title': title,
            'source_type': 'case',
            'content': content,
        }
    )

faq_topics = ['盗窃罪', '合同违约', '租赁纠纷', '侵权责任', '公司纠纷', '劳动争议']
faqs = []
for i in range(1, 21):
    topic = faq_topics[(i - 1) % len(faq_topics)]
    q = f'{topic}中，当事人应如何主张权利？第{i}问'
    a = '一般应先固定证据，再依据相关法条主张责任；如协商不成，可依法起诉或申请仲裁。'
    faqs.append(
        {
            'faq_id': f'faq_{i:03d}',
            'question': q,
            'answer': a,
            'source_type': 'faq',
        }
    )

with (root / 'statutes' / 'statutes_raw_round1.jsonl').open('w', encoding='utf-8') as f:
    for row in statutes:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

with (root / 'cases' / 'cases_raw_round1.jsonl').open('w', encoding='utf-8') as f:
    for row in cases:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

with (root / 'faqs' / 'faqs_raw_round1.jsonl').open('w', encoding='utf-8') as f:
    for row in faqs:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print({'statutes': len(statutes), 'cases': len(cases), 'faqs': len(faqs)})
