"""Create nested 96/192/384 text datasets with balanced classes and formats."""
from collections import Counter, defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1] / 'data'
TASKS = ('YAHOO','AMAZON','AGNEWS','DBPEDIA')

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(path): return {'path':str(path),'sha256':digest(path)}
def balanced(total, keys, seed=1):
    keys=sorted(keys)
    extra=set(random.Random(seed).sample(keys,total%len(keys)))
    return {key:total//len(keys)+int(key in extra) for key in keys}

def select_joint(entries, class_targets, format_targets, seed=1):
    """Integer max flow enforces class and template quotas simultaneously."""
    assert sum(class_targets.values())==sum(format_targets.values())
    pools=defaultdict(list)
    rng=random.Random(seed)
    for i,entry in enumerate(entries): pools[(entry['label'],entry['format_id'])].append(i)
    for pool in pools.values(): rng.shuffle(pool)
    residual=defaultdict(dict)
    capacity={}
    def add(a,b,n):
        if n<0: raise ValueError('Negative quota')
        residual[a][b]=n; residual[b][a]=0
        capacity[(a,b)]=n
    source,sink=('source',),('sink',)
    labels=list(class_targets);rng.shuffle(labels)
    formats=list(format_targets);rng.shuffle(formats)
    for label in labels: add(source,('class',label),class_targets[label])
    for label in labels:
        for fmt in formats:
            if pools[(label,fmt)]: add(('class',label),('format',fmt),len(pools[(label,fmt)]))
    for fmt in formats: add(('format',fmt),sink,format_targets[fmt])
    total=0
    while True:
        parent={source:None}; queue=deque([source])
        while queue and sink not in parent:
            a=queue.popleft()
            for b,n in residual[a].items():
                if n>0 and b not in parent: parent[b]=a;queue.append(b)
        if sink not in parent: break
        amount=sum(class_targets.values());node=sink
        while parent[node] is not None:
            amount=min(amount,residual[parent[node]][node]);node=parent[node]
        node=sink
        while parent[node] is not None:
            a=parent[node];residual[a][node]-=amount;residual[node][a]+=amount;node=a
        total+=amount
    if total!=sum(class_targets.values()):
        raise ValueError(f'Cannot jointly balance classes and formats: {total}/{sum(class_targets.values())}')
    indices=[]
    for (label,fmt),pool in pools.items():
        edge=(('class',label),('format',fmt))
        used=capacity.get(edge,0)-residual.get(edge[0],{}).get(edge[1],0)
        indices.extend(pool[:used])
    assert len(indices)==total==len(set(indices))
    assert Counter(entries[i]['label'] for i in indices)==Counter(class_targets)
    assert Counter(entries[i]['format_id'] for i in indices)==Counter(format_targets)
    return sorted(indices)

def main():
    old_files={str(p):digest(p) for t in TASKS for p in (ROOT/f'{t}_text').iterdir() if p.is_file()}
    for task in TASKS:
        for count in (96,384):
            for suffix in ('.jsonl','.txt','_manifest.json'):
                p=ROOT/f'{task}_text/{task}_domain_{count}{suffix}'
                if p.exists(): raise FileExistsError(p)
    prepared=[]
    for task in TASKS:
        folder=ROOT/f'{task}_text'
        parent=folder/f'{task}_domain_192.jsonl'
        additional=folder/f'{task}_domain_800.jsonl'
        manifest192=read(folder/f'{task}_domain_192_manifest.json')
        manifest800=read(folder/f'{task}_domain_800_manifest.json')
        assert digest(parent)==manifest192['output_sha256']
        assert digest(additional)==manifest800['output_sha256']
        assert manifest192['source_sha256']==manifest800['source_sha256']==digest(ROOT/task/'train.jsonl')
        rows192=[json.loads(s) for s in parent.read_text().splitlines()]
        rows800=[json.loads(s) for s in additional.read_text().splitlines()]
        assert len(rows192)==len(manifest192['examples'])==192
        assert len(rows800)==len(manifest800['examples'])==800
        assert {e['source_row_index'] for e in manifest192['examples']}.isdisjoint(e['source_row_index'] for e in manifest800['examples'])
        labels=sorted({e['label'] for e in manifest192['examples']})
        original_classes=Counter(e['label'] for e in manifest192['examples'])
        original_formats=Counter(e['format_id'] for e in manifest192['examples'])
        for count in (96,384):
            class_targets=balanced(count,labels)
            format_targets=balanced(count,list(range(1,21)))
            if count==96:
                selected=select_joint(manifest192['examples'],class_targets,format_targets)
                pieces=[(parent,rows192,manifest192,selected)]
            else:
                extra_classes={label:class_targets[label]-original_classes[label] for label in labels}
                extra_formats={fmt:format_targets[fmt]-original_formats[fmt] for fmt in range(1,21)}
                selected=select_joint(manifest800['examples'],extra_classes,extra_formats)
                pieces=[(parent,rows192,manifest192,list(range(192))),
                        (additional,rows800,manifest800,selected)]
            rows,entries=[],[]
            for path,parent_rows,parent_manifest,indices in pieces:
                for parent_index in indices:
                    row=copy.deepcopy(parent_rows[parent_index]);entry=copy.deepcopy(parent_manifest['examples'][parent_index])
                    assert hashlib.sha256(row['text'].encode()).hexdigest()==entry['text_sha256']
                    entry.update(output_row_index=len(rows),parent_file=str(path),parent_output_row_index=parent_index)
                    rows.append(row);entries.append(entry)
            assert len(rows)==len({r['text'] for r in rows})==count
            assert len({e['source_row_index'] for e in entries})==count
            assert Counter(e['label'] for e in entries)==Counter(class_targets)
            assert Counter(e['format_id'] for e in entries)==Counter(format_targets)
            sources={e['source_row_index'] for e in entries}
            parent_sources={e['source_row_index'] for e in manifest192['examples']}
            assert (sources < parent_sources) if count==96 else (parent_sources < sources)
            path=folder/f'{task}_domain_{count}.jsonl'
            encoded=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows)
            manifest=copy.deepcopy(manifest192)
            manifest.pop('per_class',None)
            for key in ('removed_parent_output_row_indices','subset_selection','parent_sampling','parent_assignment','subset_seed','derived_from'):
                manifest.pop(key,None)
            manifest.update(count=count,output_file=str(path),output_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                            per_class_counts={str(k):v for k,v in class_targets.items()},seed=1,
                            derived_from=[{**identity(p),'source_count':len(pr),'included_count':len(ix)} for p,pr,pm,ix in pieces],
                            sampling='Nested dose experiment: 96 is a subset of the original 192; 384 retains all original 192 plus 192 disjoint examples from the new 800. Seed-1 integer max flow jointly balances class and template quotas. Preserve parent text and format exactly.',
                            nesting={'reference_count':192,'relation':'subset' if count==96 else 'superset','reference':identity(parent)},
                            examples=entries)
            manifest['formatting'].pop('examples_per_format',None)
            manifest['formatting'].update(format_counts={str(k):v for k,v in format_targets.items()},
                                          assignment='Parent templates and format IDs unchanged. Jointly balanced global template and class quotas; per-class template frequencies may differ.',
                                          subset_assignment='No regeneration or paraphrasing; subset membership is the only change.')
            for label in labels: manifest['label_mapping'][str(label)]['count']=class_targets[label]
            prepared.append((task,count,rows,manifest,((path,encoded),
                              (folder/f'{task}_domain_{count}.txt','\n\n'.join(r['text'] for r in rows)+'\n'),
                              (folder/f'{task}_domain_{count}_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n'))))
    for task,count,rows,manifest,files in prepared:
        for path,body in files:
            tmp=path.with_name(path.name+'.tmp');tmp.write_text(body,encoding='utf-8');tmp.replace(path)
        assert digest(files[0][0])==manifest['output_sha256']
        print(task,count,'class counts',manifest['per_class_counts'],'format counts',manifest['formatting']['format_counts'],flush=True)
    assert all(digest(Path(path))==value for path,value in old_files.items())
    print('All old text files unchanged; verified nested 96 < 192 < 384 for all four tasks.')

if __name__=='__main__': main()
