import sys, numpy as np, onnxruntime as ort
from PIL import Image
SIZE=320; SCORE=0.25; NMS=0.45
def segment(path, session):
    img=Image.open(path).convert('RGB'); W,H=img.size
    x=np.asarray(img.resize((SIZE,SIZE),Image.BILINEAR),dtype=np.float32)/255.
    o0,o1=session.run(None,{'images':x.transpose(2,0,1)[None]})
    p=o0[0].T            # 2100 x 37 ; cols: cx cy w h score coef*32
    proto=o1[0].reshape(32,-1)
    keep=np.where(p[:,4]>SCORE)[0]
    keep=keep[np.argsort(-p[keep,4])]
    boxes=np.stack([p[:,0]-p[:,2]/2,p[:,1]-p[:,3]/2,p[:,0]+p[:,2]/2,p[:,1]+p[:,3]/2],1)
    sel=[]
    for i in keep:
        ok=True
        for j in sel:
            a,b=boxes[i],boxes[j]
            iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
            inter=iw*ih; u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            if inter/u>NMS: ok=False;break
        if ok: sel.append(i)
    mask=np.zeros((H,W),bool)
    ys=(np.arange(H)+0.5)/H; xs=(np.arange(W)+0.5)/W
    for i in sel:
        logit=(p[i,5:]@proto).reshape(80,80)
        # bilinear sample logit at output pixel centres (80-grid, align corners false)
        gy=np.clip(ys*80-0.5,0,79); gx=np.clip(xs*80-0.5,0,79)
        y0=np.floor(gy).astype(int); x0=np.floor(gx).astype(int); y1=np.minimum(y0+1,79); x1=np.minimum(x0+1,79)
        fy=(gy-y0)[:,None]; fx=(gx-x0)[None,:]
        L=(logit[y0][:,x0]*(1-fx)+logit[y0][:,x1]*fx)*(1-fy)+(logit[y1][:,x0]*(1-fx)+logit[y1][:,x1]*fx)*fy
        m=L>0
        bx=boxes[i]  # normalised? decided by SCALE
        x1b,y1b,x2b,y2b=[v*SCALE for v in bx]
        cm=np.zeros((H,W),bool)
        cm[int(max(0,y1b*H)):int(min(H,np.ceil(y2b*H))), int(max(0,x1b*W)):int(min(W,np.ceil(x2b*W)))]=True
        mask|=m&cm
    return img,mask,sel,boxes[sel] if sel else None
SCALE=1.0
if __name__=='__main__':
    s=ort.InferenceSession('QATbest.onnx',providers=['CPUExecutionProvider'])
    img,mask,sel,b=segment(sys.argv[1],s)
    print('instances',len(sel),'boxes',b,'mask px',mask.sum(),'of',mask.size)
    if len(sys.argv)>2:
        a=np.asarray(img).copy(); a[mask]=(0.5*a[mask]+[127,0,0]).astype(np.uint8); Image.fromarray(a).save(sys.argv[2])
