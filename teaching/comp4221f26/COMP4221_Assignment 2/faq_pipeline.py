"""Provided role/content encoding and deterministic response checks."""
import re


def encode_messages(tokenizer,messages,append_assistant_header=False):
    ids=[];spans=[]
    for message in messages:
        role=message['role']
        ids.extend(tokenizer.encode(role.capitalize()+':\n',add_special_tokens=False))
        start=len(ids)
        ids.extend(tokenizer.encode(message['content'],add_special_tokens=False))
        if role=='assistant':
            ids.append(tokenizer.eos_token_id)
            spans.append((start,len(ids)))
        else:
            ids.extend(tokenizer.encode('\n',add_special_tokens=False))
    if append_assistant_header:
        ids.extend(tokenizer.encode('Assistant:\n',add_special_tokens=False))
    return ids,spans


def check_response(response,expected_answer,expected_source):
    lines=response.strip().splitlines()
    format_pass=(len(lines)==2 and lines[0].startswith('Answer: ')
                 and bool(lines[0][8:].strip()) and re.fullmatch(r'Source: F[0-9]{3}',lines[1]) is not None)
    if not format_pass:return {'format_pass':False,'grounded_pass':False}
    normalize=lambda x:' '.join(x.casefold().split())
    grounded_pass=(lines[1][8:]==expected_source and normalize(lines[0][8:])==normalize(expected_answer))
    return {'format_pass':True,'grounded_pass':grounded_pass}
