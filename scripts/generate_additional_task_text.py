"""Add 800 disjoint text examples per task using the audited original 20 formats.

Uses only current QA train rows; preserves original text fields verbatim and
keeps old artifacts untouched. No model weights, training, or inference needed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('YAHOO', 'AMAZON', 'AGNEWS', 'DBPEDIA')
FIELDS = {'YAHOO': ('question_title','question_content','best_answer'),
          'AMAZON': ('title','content'), 'AGNEWS': ('text',), 'DBPEDIA': ('title','content')}
KINDS = ('a company','an educational institution','an artist','an athlete','an office holder',
         'a means of transportation','a building','a natural place','a village','an animal',
         'a plant','an album','a film','a written work')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def identity(path):
    return {'path': str(path.resolve()), 'sha256': sha(path.read_bytes())}

def content_key(task, row):
    # Compare source content, not label or wrapper, so relabeling/reformatting
    # cannot turn an old source document into a supposedly new example.
    return tuple(re.sub(r'\s+', ' ', row.get(field) or '').strip().casefold() for field in FIELDS[task])

def quote(text):
    return '“' + text + '”'

def sentence(text):
    return quote(text) + ('' if text.rstrip().endswith(('.', '?', '!', '。', '？', '！')) else '.')

def render(task, row, format_id, reference):
    variant = format_id - 1
    phrase = reference['label_mapping'][str(row['label'])]['topic_phrase']
    template = reference['formatting']['templates'][str(format_id)]
    if task == 'YAHOO':
        content = row.get('question_content')
        wrappers = reference['formatting']['question_content_wrappers']
        fields = {'topic': phrase, 'q': quote(row['question_title']), 'qs': sentence(row['question_title']),
                  'ans': sentence(row['best_answer']),
                  'details': wrappers[variant % len(wrappers)].format(content=sentence(content)) if content else ''}
    elif task == 'AMAZON':
        fields = {'stars': phrase, 'hyphen': ('one','two','three','four','five')[row['label']],
                  'title': quote(row['title']), 'body': sentence(row['content'])}
    elif task == 'AGNEWS':
        fields = {'topic': phrase, 'raw': row['text'], 'quote': quote(row['text']), 'body': sentence(row['text'])}
    else:
        fields = {'name': quote(row['title']), 'kind': KINDS[row['label']],
                  'category': phrase, 'body': sentence(row['content'])}
    text = template.format(**fields)
    for field in FIELDS[task]:
        value = row.get(field)
        if value:
            assert value in text, (task, field)
    assert phrase in text or task == 'AMAZON' and fields['hyphen']+'-star' in text
    return text

def main(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True)
    assert tokenizer.eos_token_id is not None
    data_root = ROOT / 'data'
    references = {t: read(data_root / f'{t}_text/{t}_domain_200_manifest.json') for t in TASKS}
    patterns = [value for ref in references.values() for value in ref['formatting']['templates'].values()]
    assert len(patterns) == len(set(patterns)) == 80
    old_files = {str(p): sha(p.read_bytes()) for t in TASKS for p in (data_root / f'{t}_text').iterdir() if p.is_file()}
    outputs = {}
    for task in TASKS:
        folder = data_root / f'{task}_text'
        stem = f'{task}_domain_{args.count}'
        paths = [folder / (stem + suffix) for suffix in ('.jsonl', '.txt', '_manifest.json')]
        for path in paths:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
        outputs[task] = paths
    prepared = []
    for task in TASKS:
        ref = references[task]
        source = data_root / task / 'train.jsonl'
        source_rows = read_rows(source)
        source_digest = sha(source.read_bytes())
        assert ref['source_sha256'] == source_digest
        old_indices, old_ids, old_keys, old_texts = set(), set(), set(), set()
        excluded_manifests = []
        for path in sorted((data_root / f'{task}_text').glob('*_manifest.json')):
            previous = read(path)
            if 'examples' not in previous:
                continue
            assert previous['source_sha256'] == source_digest
            excluded_manifests.append(identity(path))
            for example in previous['examples']:
                index = example['source_row_index']
                assert source_rows[index] == example['original']
                old_indices.add(index)
                old_keys.add(content_key(task, example['original']))
                if example.get('source_id') is not None:
                    old_ids.add(example['source_id'])
        for path in (data_root / f'{task}_text').glob('*.jsonl'):
            old_texts.update(row['text'] for row in read_rows(path))
        labels = sorted(map(int, ref['label_mapping']))
        allocation = random.Random(args.seed)
        extra = set(allocation.sample(labels, args.count % len(labels)))
        counts = {label: args.count // len(labels) + int(label in extra) for label in labels}
        pools = {label: [] for label in labels}
        for index, row in enumerate(source_rows):
            source_id = row.get('id', row.get('idx'))
            if index in old_indices or source_id is not None and source_id in old_ids or content_key(task,row) in old_keys:
                continue
            required = tuple(f for f in FIELDS[task] if f != 'question_content')
            if any(not isinstance(row.get(f),str) or not row[f] for f in required):
                continue
            pools[row['label']].append(index)
        sample_rng = random.Random(args.seed)
        for label in labels:
            sample_rng.shuffle(pools[label])
        ordered_labels = [label for label in labels for _ in range(counts[label])]
        sample_rng.shuffle(ordered_labels)
        format_rng = random.Random(args.seed)
        format_cycle = list(range(1,21)); format_rng.shuffle(format_cycle)
        assignments = {}
        cursor = 0
        for label in labels:
            positions = [i for i, value in enumerate(ordered_labels) if value == label]
            formats = [format_cycle[(cursor + j) % 20] for j in range(len(positions))]
            cursor += len(positions)
            format_rng.shuffle(formats)
            assignments.update(zip(positions,formats))
        rows, audit, lengths = [], [], []
        new_keys, new_ids, rejected = set(), set(), []
        for position, label in enumerate(ordered_labels):
            format_id = assignments[position]
            while True:
                if not pools[label]:
                    raise ValueError(f'{task}: insufficient unused valid rows for class {label}')
                index = pools[label].pop()
                original = source_rows[index]
                key = content_key(task,original)
                source_id = original.get('id',original.get('idx'))
                if key in new_keys or source_id is not None and source_id in new_ids:
                    rejected.append({'source_row_index':index,'reason':'Duplicate source content or ID within new samples'})
                    continue
                text = render(task,original,format_id,ref)
                length = len(tokenizer(text,add_special_tokens=False)['input_ids']) + int(tokenizer.bos_token_id is not None) + 1
                if length > args.max_length:
                    rejected.append({'source_row_index':index,'reason':'Full text exceeds token limit; no truncation','tokens':length})
                    continue
                if text in old_texts:
                    raise AssertionError('New output duplicates an old text')
                new_keys.add(key)
                if source_id is not None: new_ids.add(source_id)
                break
            lengths.append(length)
            rows.append({'text':text})
            category = ref['label_mapping'][str(label)]
            audit.append({'output_row_index':position,'source_row_index':index,'source_line_number':index+1,
                          'source_id':source_id,'label':label,'category':category['category'],
                          'category_in_text':category['topic_phrase'],'format_id':format_id,
                          'original':original,'text_sha256':sha(text.encode('utf-8')),'tokens_including_bos_eos':length})
        assert len(rows) == len(new_keys) == args.count
        assert len({row['text'] for row in rows}) == args.count
        assert {entry['source_row_index'] for entry in audit}.isdisjoint(old_indices)
        assert new_keys.isdisjoint(old_keys) and new_ids.isdisjoint(old_ids)
        assert Counter(entry['label'] for entry in audit) == Counter(counts)
        format_counts = Counter(entry['format_id'] for entry in audit)
        assert format_counts == Counter({i:args.count//20 for i in range(1,21)})
        for label in labels:
            frequencies = Counter(e['format_id'] for e in audit if e['label']==label)
            assert max(frequencies.get(i,0) for i in range(1,21))-min(frequencies.get(i,0) for i in range(1,21)) <= 1
        encoded = ''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows)
        path_jsonl,path_txt,path_manifest = outputs[task]
        formatting = dict(ref['formatting'])
        formatting.update(seed=args.seed,examples_per_format=args.count//20,
                          assignment='A reproducibly shuffled 20-format cycle allocated across classes, shuffled within each class; each format has equal global count and class-level counts differ by at most one.',
                          format_counts={str(i):format_counts[i] for i in range(1,21)})
        mapping = {k:{**value,'count':counts[int(k)]} for k,value in ref['label_mapping'].items()}
        manifest = {'source_file':str(source),'source_split':'train','source_sha256':source_digest,
                    'output_file':str(path_jsonl),'output_sha256':sha(encoded.encode('utf-8')),
                    'seed':args.seed,'count':args.count,'per_class_counts':{str(k):v for k,v in counts.items()},
                    'sampling':'Stratified sampling without replacement, seed 1. Exclude all previously exported source indices, IDs and normalized original-field content. Classes differ by at most one sample. All content preserved verbatim.',
                    'transformation':ref['transformation'],'output_fields':['text'],'label_mapping':mapping,'formatting':formatting,
                    'template_reference':identity(data_root/f'{task}_text/{task}_domain_200_manifest.json'),
                    'excluded_previous_manifests':excluded_manifests,'excluded_previous_unique_source_rows':len(old_indices),
                    'length_check':{'tokenizer':str(ROOT/'Llama-3.1-8B-Instruct'),'max_allowed_tokens':args.max_length,
                                    'count_includes':'BOS if configured, full raw text and EOS','truncated_examples':0,
                                    'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths)},
                    'rejected_candidates':rejected,'examples':audit}
        prepared.append((task,rows,manifest,((path_jsonl,encoded),(path_txt,'\n\n'.join(row['text'] for row in rows)+'\n'),
                        (path_manifest,json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n'))))
    # Validate all tasks before writing any outputs; never touch the old exports.
    assert all(sha(Path(path).read_bytes())==digest for path,digest in old_files.items())
    for task,rows,manifest,files in prepared:
        for path,body in files:
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_text(body,encoding='utf-8'); temporary.replace(path)
        saved=read_rows(files[0][0])
        assert saved==rows and sha(files[0][0].read_bytes())==manifest['output_sha256']
        for row,entry in zip(saved,manifest['examples']):
            assert set(row)=={'text'} and sha(row['text'].encode('utf-8'))==entry['text_sha256']
            assert all(not entry['original'].get(f) or entry['original'][f] in row['text'] for f in FIELDS[task])
        print(json.dumps({'task':task,'count':len(rows),'per_class_counts':manifest['per_class_counts'],
                          'formats':20,'per_format':args.count//20,'old_sample_overlap':0,
                          'length_check':manifest['length_check'],'rejected_candidates':len(manifest['rejected_candidates']),
                          'output_file':manifest['output_file']},ensure_ascii=False),flush=True)
    assert all(sha(Path(path).read_bytes())==digest for path,digest in old_files.items())
    print('Verified: all old artifacts unchanged; 3200 additional text examples with no old-source overlap.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=800)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--max_length',type=int,default=1536)
    args=parser.parse_args()
    if args.count<=0 or args.count%20 or args.max_length<2:
        raise ValueError('Count must be a positive multiple of 20, max_length >= 2')
    main(args)
