import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json
from pathlib import Path
import cv2,numpy as np,trimesh
root=Path(__file__).resolve().parent
manifest=json.loads((root/'input_manifest.json').read_text());stats=[]
for name in manifest['selected_images']:
 p=root/'results'/Path(name).stem
 mask=cv2.imread(str(p/'mask.png'),0)>0
 d=cv2.imread(str(p/'depth.exr'),cv2.IMREAD_UNCHANGED)
 pts=cv2.imread(str(p/'points.exr'),cv2.IMREAD_UNCHANGED)
 assert d.shape==mask.shape and pts.shape==(*d.shape,3)
 assert list(d.shape)==manifest['shapes'][Path(name).name][:2]
 assert mask.any() and np.isfinite(d[mask]).all() and (d[mask]>0).all()
 assert np.isfinite(pts[mask]).all()
 pc=trimesh.load(p/'pointcloud.ply',process=False)
 assert len(pc.vertices)>0 and np.isfinite(pc.vertices).all()
 glb=trimesh.load(p/'mesh.glb',process=False)
 assert len(glb.geometry)>0
 r={'image':Path(name).name,'shape':list(d.shape),'valid_fraction':float(mask.mean()),'depth_p05_p50_p95':np.percentile(d[mask],[5,50,95]).tolist(),'ply_vertices':len(pc.vertices),'glb_geometries':len(glb.geometry)}
 stats.append(r);print(json.dumps(r),flush=True)
(root/'validation.json').write_text(json.dumps({'images_checked':len(stats),'all_passed':True,'results':stats},indent=2))
# A compact overview of every tested camera and time sample.
rows=[]
for frame in manifest['selected_frames']:
 cols=[]
 for name in manifest['selected_images']:
  if not Path(name).stem.startswith(frame+'_'):continue
  p=root/'results'/Path(name).stem
  color=cv2.imread(str(p/'image.jpg'));depth=cv2.imread(str(p/'depth_vis.png'))
  color=cv2.resize(color,(256,160));depth=cv2.resize(depth,(256,160))
  cv2.putText(color,Path(name).stem,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
  cols.append(np.concatenate([color,depth],axis=0))
 rows.append(np.concatenate(cols,axis=1))
cv2.imwrite(str(root/'overview.jpg'),np.concatenate(rows,axis=0))
