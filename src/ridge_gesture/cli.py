from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(prog="ridge-gesture"); s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("annotate"); a.add_argument("--transcripts",required=True); a.add_argument("--output",required=True); a.add_argument("--external-annotations")
    b=s.add_parser("build-rules"); b.add_argument("--records",required=True); b.add_argument("--annotations",required=True); b.add_argument("--output",required=True); b.add_argument("--sbert",default="all-MiniLM-L6-v2")
    t=s.add_parser("train"); t.add_argument("--pairs",required=True); t.add_argument("--output",required=True); t.add_argument("--epochs",type=int,default=50); t.add_argument("--batch-size",type=int,default=64); t.add_argument("--seed",type=int,default=0)
    r=s.add_parser("retrieve"); r.add_argument("--rules",required=True); r.add_argument("--checkpoint",required=True); r.add_argument("--text",required=True); r.add_argument("--threshold",type=float,default=.72); r.add_argument("--output",required=True); r.add_argument("--sbert",default="all-MiniLM-L6-v2")
    g=s.add_parser("eval-gca"); g.add_argument("--reference",required=True); g.add_argument("--candidate",required=True); g.add_argument("--text-clusters",type=int,default=100); g.add_argument("--gesture-clusters",type=int,default=20)
    x=p.parse_args(); {"annotate":annotate,"build-rules":build_rules,"train":train,"retrieve":retrieve,"eval-gca":eval_gca}[x.cmd](x)

def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def dump(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")

def annotate(a):
    from .pipeline import heuristic_phrases
    records=lines(a.transcripts); external=json.loads(Path(a.external_annotations).read_text(encoding="utf-8")) if a.external_annotations else {}
    dump(a.output,[{"record_id":r["record_id"],"phrases":external.get(r["record_id"],heuristic_phrases(r["text"])),"annotator":"external" if r["record_id"] in external else "heuristic"} for r in records])

def build_rules(a):
    from sentence_transformers import SentenceTransformer
    from .pipeline import align_phrase
    rec={r["record_id"]:r for r in lines(a.records)}; rows=[]
    for item in lines(a.annotations):
        for phrase in item["phrases"]:
            start,end=align_phrase(phrase,rec[item["record_id"]]["words"]); rows.append({"phrase":phrase,"gesture_id":f'{item["record_id"]}:{start}-{end}',"record_id":item["record_id"],"start_frame":start,"end_frame":end})
    emb=SentenceTransformer(a.sbert).encode([r["phrase"] for r in rows],normalize_embeddings=True)
    for r,e in zip(rows,emb): r["embedding"]=e.tolist()
    dump(a.output,rows)

def train(a):
    import torch
    from .model import TextMotionModel,contrastive
    d=np.load(a.pairs); te=d["text_embeddings"].astype("float32"); mo=d["motion"].astype("float32"); ids=np.asarray(d["ids"])
    if len(te)!=len(mo) or len(ids)!=len(te) or len(te)<2: raise ValueError("training requires at least two aligned text/motion pairs and IDs")
    if a.epochs<1 or a.batch_size<2: raise ValueError("epochs must be positive and batch size must be at least two")
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); model=TextMotionModel(te.shape[-1],mo.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    for epoch in range(a.epochs):
        order=rng.permutation(len(te)); total=0.; seen=0
        for st in range(0,len(order),a.batch_size):
            ix=order[st:st+a.batch_size]
            if len(ix)<2: continue
            zt,zm=model(torch.from_numpy(te[ix]),torch.from_numpy(mo[ix])); loss=contrastive(zt,zm); opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ix); seen+=len(ix)
        print(json.dumps({"epoch":epoch+1,"loss":total/seen}))
    model.eval()
    with torch.no_grad(): _,lat=model(torch.from_numpy(te),torch.from_numpy(mo))
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); torch.save({"state":model.state_dict(),"text_dim":te.shape[-1],"motion_dim":mo.shape[-1],"ids":ids.tolist(),"motion_latents":lat.cpu()},output)

def retrieve(a):
    import torch
    from sentence_transformers import SentenceTransformer
    from .model import TextMotionModel
    from .pipeline import hybrid_retrieve
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True); model=TextMotionModel(ck["text_dim"],ck["motion_dim"]); model.load_state_dict(ck["state"]); model.eval(); sbert=SentenceTransformer(a.sbert); rules=lines(a.rules); latent=ck["motion_latents"].cpu().numpy().astype("float32"); ids=[str(x) for x in ck["ids"]]
    def enc(text):
        with torch.no_grad():
            z=model.text(torch.from_numpy(sbert.encode([text],normalize_embeddings=True).astype("float32"))).numpy()[0]
        return z/max(np.linalg.norm(z),1e-8)
    result=hybrid_retrieve(a.text,rules,lambda x:sbert.encode(x,normalize_embeddings=True),a.threshold,latent,ids,enc); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8")

def eval_gca(a):
    from .pipeline import GCA
    ref=np.load(a.reference); cand=np.load(a.candidate); metric=GCA(a.text_clusters,a.gesture_clusters).fit(ref["text_embeddings"],ref["motion_embeddings"]); print(json.dumps({"gca":metric.score(cand["text_embeddings"],cand["motion_embeddings"]),"fit_samples":len(ref["text_embeddings"]),"evaluation_samples":len(cand["text_embeddings"])}))

if __name__=="__main__": main()
