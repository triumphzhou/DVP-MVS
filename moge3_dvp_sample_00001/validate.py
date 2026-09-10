from pathlib import Path
import json,struct,os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import numpy as np,cv2,trimesh
root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());rows=[]
def read(p):
 with p.open('rb') as f:
  v,h,w,t=struct.unpack('4i',f.read(16));a=np.frombuffer(f.read(),np.float32)
 assert v==1
 return a.reshape(h,w,3) if t==21 else a.reshape(h,w)
for r in m['images']:
 d=read(root/'scene/APD'/f"{r['id']:08d}"/'depths.dmb');prior=read(root/'scene/metric_prior'/f"{r['id']:08d}.dmb")
 assert d.shape==(640,960) and np.isfinite(d).all()
 v=d>0;assert v.any();common=v&(prior>0)
 rows.append({'id':r['id'],'image':r['image'],'valid_fraction':float(v.mean()),'median_depth':float(np.median(d[v])),'median_abs_change_from_prior':float(np.median(np.abs(d[common]-prior[common])))})
p=root/'scene/APD/APD.ply';pc=trimesh.load(p,process=False);assert len(pc.vertices)>0 and np.isfinite(pc.vertices).all()
out={'passed':True,'views':rows,'fused_points':len(pc.vertices),'pointcloud':str(p)}
(root/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
# Shared color range for prior/result comparisons; black is invalid.
vis=[]
for r in m['images']:
 i=r['id'];im=cv2.imread(str(root/'scene/images'/f'{i:08d}.jpg'))
 maps=[read(root/'scene/metric_prior'/f'{i:08d}.dmb'),read(root/'scene/APD'/f'{i:08d}'/'depths.dmb')]
 panels=[cv2.resize(im,(320,213))]
 for a in maps:
  gray=(np.clip(np.log1p(a)/np.log(121),0,1)*255).astype('uint8');c=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);c[a<=0]=0;panels.append(cv2.resize(c,(320,213)))
 cv2.putText(panels[0],r['image'],(6,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
 vis.append(np.concatenate(panels,axis=1))
cv2.imwrite(str(root/'comparison.jpg'),np.concatenate(vis,axis=0))
