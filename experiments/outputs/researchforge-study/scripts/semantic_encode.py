"""Pinned public MiniLM encoder, local CPU only, streamed resumable TRAIN embeddings.

The timing mode does not access relevance labels. Full mode reads only TRAIN text.
No remote inference and no Hugging Face files outside the task workspace.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'work'
os.environ['HF_HOME']=str(WORK/'hf-cache')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
MODEL=WORK/'models/all-MiniLM-L6-v2'
REPO='sentence-transformers/all-MiniLM-L6-v2'
DEST=WORK/'indices/prescience-minilm'

def download():
    from huggingface_hub import HfApi,hf_hub_download
    MODEL.mkdir(parents=True,exist_ok=True)
    manifest=MODEL/'download_manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    info=HfApi().model_info(REPO)
    names={f.rfilename for f in info.siblings}
    selected=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','README.md','sentence_bert_config.json']
    result=dict(repository=REPO,revision=info.sha,files={})
    for name in selected:
        if name not in names:continue
        if shutil.disk_usage(WORK).free<1200*1024**2:raise OSError('Insufficient disk reserve')
        p=Path(hf_hub_download(REPO,name,revision=info.sha,local_dir=MODEL))
        with p.open('rb') as f:result['files'][name]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    manifest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True);return result

class Encoder:
    def __init__(self,threads=2):
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.model=AutoModel.from_pretrained(MODEL,local_files_only=True,use_safetensors=True).eval()
    def encode(self,texts):
        torch=self.torch
        tokens=self.tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.inference_mode():
            output=self.model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1).to(output.dtype)
            embedding=(output*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            embedding=torch.nn.functional.normalize(embedding,p=2,dim=1)
        return embedding.numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');ap.add_argument('--benchmark',type=int,default=0)
    ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--threads',type=int,default=2);args=ap.parse_args()
    if args.download:download();return
    import numpy as np
    import pyarrow.parquet as pq
    import psutil
    source=WORK/'data/prescience/train.parquet'
    model_manifest=json.loads((MODEL/'download_manifest.json').read_text())
    config=dict(repository=REPO,revision=model_manifest['revision'],max_length=256,pooling='attention-mask mean + L2 normalization',
        source_sha256=json.loads(source.with_suffix('.parquet.manifest.json').read_text())['sha256'],
        batch_size=args.batch_size,threads=args.threads,stored_dtype='float16',dimensions=384)
    encoder=Encoder(args.threads)
    if args.benchmark:
        rows=next(pq.ParquetFile(source).iter_batches(batch_size=args.benchmark,columns=['title','abstract'])).to_pylist()
        texts=[(r['title'] or '')+' '+(r['abstract'] or '') for r in rows]
        encoder.encode(texts[:min(2,len(texts))])
        began=time.perf_counter();chunks=[]
        for i in range(0,len(texts),args.batch_size):chunks.append(encoder.encode(texts[i:i+args.batch_size]))
        seconds=time.perf_counter()-began;x=np.concatenate(chunks)
        result=dict(config=config,documents=len(texts),seconds=seconds,docs_per_second=len(texts)/seconds,
            estimated_train_hours=373716/(len(texts)/seconds)/3600,process_rss_bytes=psutil.Process().memory_info().rss,
            max_norm_error=float(np.max(np.abs(np.linalg.norm(x,axis=1)-1))),finite=bool(np.isfinite(x).all()),
            caveat='Short timing sample; corpus length/system contention may change throughput.')
        dest=ROOT/'outputs/researchforge-study/results/semantic-timing';dest.mkdir(parents=True,exist_ok=True)
        (dest/f'minilm-b{args.batch_size}-t{args.threads}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True);return
    DEST.mkdir(parents=True,exist_ok=True)
    state_path=DEST/'progress.json';data_path=DEST/'embeddings.f16'
    if (DEST/'manifest.json').exists():
        print('Completed embeddings already exist',flush=True);return
    if state_path.exists():
        state=json.loads(state_path.read_text());assert state['config']==config,'Configuration differs; do not mix embeddings'
        completed=state['completed_rows']
    else:
        completed=0
        state=dict(config=config,completed_rows=0)
    total=pq.ParquetFile(source).metadata.num_rows
    needed=(total-completed)*384*2
    if shutil.disk_usage(WORK).free-needed<1024**3:raise OSError('Embedding output would leave less than1 GiB free')
    began=time.perf_counter();initial=completed;seen=0
    # Truncate any uncommitted tail left by interruption, keeping acknowledged rows.
    with data_path.open('r+b' if data_path.exists() else 'w+b') as out:
        out.truncate(completed*384*2);out.seek(0,2)
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=['title','abstract']):
            if seen+len(batch)<=completed:seen+=len(batch);continue
            rows=batch.to_pylist();skip=max(0,completed-seen);rows=rows[skip:]
            texts=[(r['title'] or '')+' '+(r['abstract'] or '') for r in rows]
            for i in range(0,len(texts),args.batch_size):
                x=encoder.encode(texts[i:i+args.batch_size])
                assert np.isfinite(x).all()
                x.astype(np.float16).tofile(out);completed+=len(x)
            out.flush();os.fsync(out.fileno())
            state=dict(config=config,completed_rows=completed,total_rows=total,
                session_seconds=time.perf_counter()-began,session_docs_per_second=(completed-initial)/max(time.perf_counter()-began,.001),
                worker_pid=os.getpid(),available_ram_bytes=psutil.virtual_memory().available)
            temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(state_path)
            print(json.dumps(state),flush=True)
            if shutil.disk_usage(WORK).free<1024**3:raise OSError('Low disk reserve; resumable checkpoint saved')
            seen+=len(batch)
    assert completed==total
    with data_path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    state.update(sha256=digest,bytes=data_path.stat().st_size,status='complete',row_order='same as train.parquet and prescience-hash18 ids.json')
    (DEST/'manifest.json').write_text(json.dumps(state,indent=2));print('Completed',json.dumps(state),flush=True)

if __name__=='__main__':main()
