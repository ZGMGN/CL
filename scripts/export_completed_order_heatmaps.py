"""Export a fixed snapshot of completed QA-only order accuracy matrices."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import zipfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT/'results/standard_cl_all_orders_seed_1_bs8_ga2'


def draw(ax, tasks, matrix, compact=False):
    im = ax.imshow(matrix*100, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(4), tasks, fontsize=11 if compact else 14)
    rows = ['M0 base']+[f'M{i+1} after {task}' for i,task in enumerate(tasks)]
    ax.set_yticks(range(5), rows, fontsize=10 if compact else 13)
    ax.set_xlabel('Evaluation task', fontsize=11 if compact else 14)
    for i in range(5):
        for j in range(4):
            ax.text(j,i,f'{matrix[i,j]*100:.2f}',ha='center',va='center',
                    color='white' if matrix[i,j]>=.65 else '#222222',fontsize=11 if compact else 16)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.2)
    ax.tick_params(which='minor',bottom=False,left=False)
    return im


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = QUEUE/f"completed_heatmaps_{now:%Y%m%d_%H%M%S}"
    out.mkdir(); (out/'individual').mkdir(); (out/'source_matrices').mkdir()
    snapshot=json.loads((QUEUE/'all_orders_summary.json').read_text())
    entries=[x for x in snapshot['orders'] if x.get('complete')]
    validated=[]
    for entry in entries:
        source=Path(entry['accuracy_matrix']); payload=json.loads(source.read_text())
        assert payload.get('complete') and payload['tasks']==entry['tasks']
        matrix=np.array(payload['accuracy'],dtype=float)
        assert matrix.shape==(5,4) and np.isfinite(matrix).all() and ((matrix>=0)&(matrix<=1)).all()
        assert np.allclose(matrix,np.array(entry['accuracy']),atol=1e-12,rtol=0)
        validated.append((entry,matrix))
    assert len(validated)==snapshot['complete_orders']
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none',
                         'font.size':13,'figure.facecolor':'white'})
    manifest=[]; numbers=[]
    with PdfPages(out/'all_completed_accuracy_heatmaps.pdf') as pdf:
        for index,(entry,matrix) in enumerate(validated,1):
            tasks=entry['tasks']; slug='_'.join(tasks); stem=f'{index:02d}_{slug}'
            fig,ax=plt.subplots(figsize=(12,8.5))
            im=draw(ax,tasks,matrix)
            ax.set_title(f"Completed order {index}/{len(validated)}\n"+' → '.join(tasks),fontsize=17,pad=20)
            fig.colorbar(im,ax=ax,fraction=.04,pad=.035,label='Accuracy (%)')
            fig.text(.02,.025,f'QA-only | seed 1 | bs8, accumulation 2 | snapshot {now:%Y-%m-%d %H:%M} Asia/Shanghai\n'
                     'Each row evaluates the base model or a stage adapter on all four tasks. Shared scale: 0–100%.',fontsize=10)
            fig.tight_layout(rect=(0,.09,1,1),pad=1.5)
            fig.savefig(out/'individual'/f'{stem}.png',dpi=200)
            fig.savefig(out/'individual'/f'{stem}.pdf')
            pdf.savefig(fig);plt.close(fig)
            shutil.copy2(entry['accuracy_matrix'],out/'source_matrices'/f'{stem}.json')
            manifest.append({'page':index,'tasks':tasks,'run_dir':entry['run_dir'],'source':entry['accuracy_matrix'],
                             'figure':f'individual/{stem}.pdf'})
            for stage in range(5):
                for col,task in enumerate(tasks):
                    numbers.append({'order':slug,'model':f'M{stage}','last_trained_task':None if stage==0 else tasks[stage-1],
                                    'evaluation_task':task,'accuracy':float(matrix[stage,col])})
    # Four orders per overview page keep all task and stage labels legible.
    with PdfPages(out/'accuracy_heatmaps_overview.pdf') as pdf:
        for start in range(0,len(validated),4):
            fig,axes=plt.subplots(2,2,figsize=(20,14),layout='constrained')
            for offset,ax in enumerate(axes.flat):
                if start+offset>=len(validated):
                    ax.set_visible(False);continue
                entry,matrix=validated[start+offset]
                im=draw(ax,entry['tasks'],matrix,compact=True)
                ax.set_title(f"{start+offset+1:02d}. "+' → '.join(entry['tasks']),fontsize=13,pad=12)
            fig.suptitle(f'Completed QA-only orders | accuracy (%) | {now:%Y-%m-%d %H:%M} Asia/Shanghai',fontsize=17)
            fig.colorbar(im,ax=list(axes.flat),fraction=.02,pad=.02,label='Accuracy (%)')
            fig.savefig(out/f'overview_{start//4+1:02d}.png',dpi=170)
            pdf.savefig(fig);plt.close(fig)
    with (out/'accuracy_values.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(numbers[0]));writer.writeheader();writer.writerows(numbers)
    (out/'manifest.json').write_text(json.dumps({'created_at':now.isoformat(),'completed_orders':len(validated),
                                               'total_orders':24,'orders':manifest},ensure_ascii=False,indent=2)+'\n')
    text=f'# 已完成无 text_mix order 的准确率 heatmap\n\n快照时间：{now:%Y-%m-%d %H:%M:%S}（北京时间）。共 {len(validated)}/24 个 order。\n\n'
    text+='all_completed_accuracy_heatmaps.pdf 每页一个 order；accuracy_heatmaps_overview.pdf 每页四个 order。\n\n'
    text+='纵轴：M0 为 base，M1–M4 为依次训练各任务后的 adapter；横轴为评估任务。所有图均为完整 5×4 矩阵，统一 0–100% 色标。\n\n'
    text+='只汇总快照中训练与准确率评估均已完成的 order，不重新运行评估。individual/ 保存各 order 的独立 PDF 和 PNG，source_matrices/ 保存原始矩阵；accuracy_values.csv 为长表数据。\n\n'
    for item in manifest:
        text+=f"{item['page']}. {' → '.join(item['tasks'])}\n"
    (out/'README.md').write_text(text)
    with zipfile.ZipFile(out/'completed_order_heatmaps.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix!='.zip':archive.write(path,path.relative_to(out))
    print(json.dumps({'output':str(out),'completed_orders':len(validated)},ensure_ascii=False))


if __name__=='__main__':main()
