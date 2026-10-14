"""Provided pipeline. Uses student record normalization/validation functions."""
import json
from pathlib import Path
from student_tasks import deduplicate_records,normalize_text

SYSTEM='Use the supplied fictional campus fact. Reply exactly two lines: Answer: followed by the answer, then Source: followed by its fact ID. Do not add any other text.'


def build_data(rows):
    train=[]
    for i,row in enumerate(rows):
        fact=f'Fact [{row["id"]}]: {row["question"]} {row["answer"]}'
        for j,intro in enumerate(['Please answer: ','I need this information: ','Could you tell me: ']):
            messages=[{'role':'system','content':SYSTEM}]
            if j==1:
                messages += [{'role':'user','content':'Can you follow the two-line response format?'},
                             {'role':'assistant','content':'Yes, I can use that format.'}]
            messages += [{'role':'user','content':fact+'\nQuestion: '+intro+row['question']},
                         {'role':'assistant','content':f'Answer: {row["answer"]}\nSource: {row["id"]}'}]
            train.append({'id':row['id']+f'-train{j}','messages':messages})
    evaluation=[]
    for row in rows[::5][:20]:
        fact=f'Fact [{row["id"]}]: {row["question"]} {row["answer"]}'
        messages=[{'role':'system','content':SYSTEM},
                  {'role':'user','content':fact+'\nQuestion: I am checking this service. '+row['question']}]
        evaluation.append({'id':row['id']+'-eval','messages':messages,
                           'expected_answer':row['answer'],'expected_source':row['id']})
    return train,evaluation


def main():
    if not Path('raw_faq.jsonl').is_file():
        raise SystemExit('Run from the extracted assignment folder containing raw_faq.jsonl.')
    raw=[json.loads(x) for x in Path('raw_faq.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    normalized=[]
    import re
    for r in raw:
        if isinstance(r,dict) and all(isinstance(r.get(k),str) for k in ('id','question','answer')):
            v={k:normalize_text(r[k]) for k in ('id','question','answer')}
            if re.fullmatch(r'F[0-9]{3}',v['id']) and v['question'] and v['answer']:normalized.append(v)
    clean=deduplicate_records(raw)
    if len(clean)!=100:
        raise SystemExit(f'Expected 100 retained facts; got {len(clean)}. Check Part A before training.')
    train,evaluation=build_data(clean)
    Path('data').mkdir(exist_ok=True)
    for name,records in [('clean_faq',clean),('train',train),('eval',evaluation)]:
        Path(f'data/{name}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    counts={'raw':len(raw),'valid_before_dedup':len(normalized),'unique':len(clean),
            'training_examples':len(train),'evaluation_prompts':len(evaluation)}
    Path('data/counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
