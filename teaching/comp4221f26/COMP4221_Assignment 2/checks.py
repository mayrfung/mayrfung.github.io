"""Public COMP4221 checks for the distinct release. Run with python checks.py."""
import argparse
import sys
import torch
import torch.nn.functional as F
from student_tasks import normalize_text,deduplicate_records,nucleus_distribution,final_response_features,conversation_balanced_loss
from faq_pipeline import check_response


def check_a():
    assert normalize_text('  ＦＡＱ\t&amp;  notes\n')=='FAQ & notes'
    records=[{'id':'F001','question':'  Hello?  ','answer':' A. '},
             {'id':'F001','question':'HELLO?','answer':'A.'},
             {'id':'bad','question':'Hello?','answer':'A.'},
             {'id':'F002','question':'','answer':'A.'},
             {'id':'F003','question':'Other?','answer':'B.'}]
    assert deduplicate_records(records)==[{'id':'F001','question':'Hello?','answer':'A.'},
                                           {'id':'F003','question':'Other?','answer':'B.'}]


def check_b():
    logits=torch.log(torch.tensor([.5,.3,.15,.05],dtype=torch.float64))
    probs=nucleus_distribution(logits,1.,.7)
    assert torch.allclose(probs.double(),torch.tensor([.625,.375,0,0],dtype=torch.float64),atol=1e-6)
    assert torch.allclose(nucleus_distribution(logits,1.,1.).double(),logits.softmax(0),atol=1e-6)
    assert (nucleus_distribution(logits,1.,.01)>0).sum()==1
    for t,p in [(0,.8),(-1,.8),(1,0),(1,1.1)]:
        try:nucleus_distribution(logits,t,p)
        except (ValueError,TypeError):pass
        else:raise AssertionError('Invalid nucleus parameters were accepted')


def check_c1():
    ids=list(range(10,20)); spans=[(2,4),(7,10)]
    feature=final_response_features(ids,spans,12,0)
    assert feature['labels'].tolist()==[-100]*7+[17,18,19]+[-100]*2
    assert feature['attention_mask'].tolist()==[1]*10+[0]*2
    assert feature['input_ids'].tolist()==ids+[0,0]
    assert final_response_features(ids,spans,5,0)['labels'].tolist()==[-100]*5
    assert final_response_features(ids,spans,8,0)['labels'].tolist()==[-100]*7+[17]
    assert final_response_features(ids,[],12,0)['labels'].tolist()==[-100]*12


def check_c2():
    torch.manual_seed(123)
    labels=torch.tensor([[0,1,2,3],[0,-100,1,-100],[0,-100,-100,-100]])
    scores=torch.randn(3,4,5,requires_grad=True)
    epsilon=.1
    tokens=F.cross_entropy(scores[:,:-1].transpose(1,2),labels[:,1:],
                           ignore_index=-100,reduction='none',label_smoothing=epsilon)
    valid=labels[:,1:]!=-100;counts=valid.sum(1)
    expected=(tokens.sum(1)[counts>0]/counts[counts>0]).mean()
    actual=conversation_balanced_loss(scores,labels,epsilon)
    assert actual.ndim==0 and torch.allclose(actual,expected,atol=1e-6)
    actual.backward();assert torch.isfinite(scores.grad).all()
    empty=torch.full_like(labels,-100)
    zero=conversation_balanced_loss(scores,empty,.1)
    assert zero.item()==0;zero.backward()


def check_d():
    assert check_response('Answer: 3 days.\nSource: F001','3 days.','F001')=={'format_pass':True,'grounded_pass':True}
    assert check_response('Answer: 3 days.\nSource: F002','3 days.','F001')=={'format_pass':True,'grounded_pass':False}
    assert not check_response('Here is your answer: 3 days.','3 days.','F001')['format_pass']


def main():
    parser=argparse.ArgumentParser(description='CPU checks. Select a part while other TODOs are unfinished.')
    parser.add_argument('--part',choices=['A','B','C1','C2','scorer','all'],default='all')
    args=parser.parse_args()
    checks={'A':check_a,'B':check_b,'C1':check_c1,'C2':check_c2,'scorer':check_d}
    failed=[]
    for part,check in checks.items():
        if args.part not in ('all',part):continue
        try:check()
        except NotImplementedError:
            print(f'{part}: TODO not implemented yet');failed.append(part)
        except Exception as exc:
            import traceback
            print(f'{part}: FAILED ({type(exc).__name__}: {exc})')
            traceback.print_exc();failed.append(part)
        else:print(f'{part}: PASS')
    if failed:
        print('Fix or finish the listed parts, then rerun. Other parts can be checked separately.')
        raise SystemExit(1)
    print('Selected checks passed. These are examples, not an exhaustive grading suite.')
    print('Scorer checks test scoring examples only; they do not complete or validate your Part D experiment.')


if __name__=='__main__':main()
