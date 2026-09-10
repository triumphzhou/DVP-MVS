import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import json,struct
import cv2,numpy as np
root=Path(__file__).resolve().parent
source=Path('/mnt/nuplan/l3data-reconstruction-bingxing/samples/sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/converted')
moge=root.parent/'moge3_sample_00001/results';scene=root/'scene';w,h=960,640
for d in ['images','cams','metric_prior']:(scene/d).mkdir(parents=True,exist_ok=True)
labels=[f'{t:06d}_{c}' for t in [0,50,100] for c in ['00','01','10']]
def dmb(path,arr):
 arr=np.ascontiguousarray(arr,dtype=np.float32);typ=5 if arr.ndim==2 else 21
 path.write_bytes(struct.pack('4i',1,*arr.shape[:2],typ)+arr.tobytes())
records=[];cams=[];depths=[]
for i,label in enumerate(labels):
 im=cv2.imread(str(source/'images'/f'{label}.jpg'));H,W=im.shape[:2]
 K=np.loadtxt(source/'intrinsics'/f'{label}.txt');T=np.loadtxt(source/'ego_pose'/f'{label}.txt');E=np.linalg.inv(T)
 d=cv2.imread(str(moge/label/'depth.exr'),cv2.IMREAD_UNCHANGED);mask=cv2.imread(str(moge/label/'mask.png'),0)>0
 lidar=np.load(source/'lidar_depth'/f'{label}.npy',allow_pickle=True).item();ld=np.zeros_like(d);ld[lidar['mask']]=lidar['value']
 valid=mask & np.isfinite(d)&(d>0)&(ld>2)&(ld<80)
 yy,xx=np.indices(d.shape);calib=valid & (yy%2==0)
 ratios=ld[calib]/d[calib];assert len(ratios)>100
 scale=float(np.median(ratios));depth=np.where(mask & np.isfinite(d)&(d>0),d*scale,0).astype(np.float32)
 # A per-view scalar calibrates predicted depth to known camera translation units.
 K[0]*=w/W;K[1]*=h/H
 depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST);depth[(depth<.5)|(depth>120)]=0
 ys,xs=np.indices((h,w));xyz=np.stack([(xs-K[0,2])*depth/K[0,0],(ys-K[1,2])*depth/K[1,1],depth],-1)
 normal=np.cross(np.gradient(xyz,axis=1),np.gradient(xyz,axis=0));norm=np.linalg.norm(normal,axis=-1,keepdims=True);normal/=np.maximum(norm,1e-8)
 flip=np.sum(normal*xyz,axis=-1)>0;normal[flip]*=-1
 normal=normal@E[:3,:3] # camera normal -> world normal, row-vector convention
 normal[depth==0]=0
 dmb(scene/'metric_prior'/f'{i:08d}.dmb',depth);dmb(scene/'metric_prior'/f'{i:08d}_normal.dmb',normal)
 cv2.imwrite(str(scene/'images'/f'{i:08d}.jpg'),cv2.resize(im,(w,h)))
 text='extrinsic\n'+'\n'.join(' '.join(map(str,row)) for row in E)+'\n\nintrinsic\n'+'\n'.join(' '.join(map(str,row)) for row in K)+'\n\n0.8333333333 0.25 397 100\n'
 (scene/'cams'/f'{i:08d}_cam.txt').write_text(text)
 records.append({'id':i,'image':label,'moge_scale_multiplier':scale,'lidar_calibration_pixels':int(calib.sum()),'valid_prior_fraction':float((depth>0).mean())});cams.append((K,T,E));depths.append(depth)
# Rank source views by projected coverage of valid reference depth samples.
lines=[str(len(labels))]
for i,(K,T,E) in enumerate(cams):
 y,x=np.mgrid[8:h-8:16,8:w-8:16];z=depths[i][y,x];good=z>0
 pts=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],-1)[good];world=pts@T[:3,:3].T+T[:3,3]
 scores=[]
 for j,(Kj,Tj,Ej) in enumerate(cams):
  if i==j:continue
  q=world@Ej[:3,:3].T+Ej[:3,3];uv=q@Kj.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
  inside=(q[:,2]>0)&(uv[:,0]>3)&(uv[:,0]<w-3)&(uv[:,1]>3)&(uv[:,1]<h-3)
  score=float(inside.mean());
  if score>.03:scores.append((j,score))
 scores.sort(key=lambda t:t[1],reverse=True);assert len(scores)>=2,(i,scores)
 lines.extend([str(i),str(len(scores))+' '+' '.join(f'{j} {s:.6f}' for j,s in scores)]);records[i]['source_views']=scores
(scene/'pair.txt').write_text('\n'.join(lines)+'\n')
(root/'manifest.json').write_text(json.dumps({'source':str(source),'size':[w,h],'prior':'MoGe3 depth, per-view median LiDAR scale; normals derived with calibrated K','images':records},indent=2))
print(json.dumps(records,indent=2))
