import sys,numpy as np,onnxruntime as ort
sys.path.insert(0,sys.argv[0].rsplit('/',1)[0]); import ref
s=ort.InferenceSession('QATbest.onnx',providers=['CPUExecutionProvider'])
for n in ['Demo_0','Demo_1']:
    img,m,sel,b=ref.segment(f'Wound kit Final images/{n}.jpg',s)
    j=np.frombuffer(open(f'{sys.argv[1]}/{n.lower().replace("_","")}.mask','rb').read(),np.uint8).reshape(m.shape).astype(bool)
    print(n,'py px',m.sum(),'js px',j.sum(),'IoU',round((m&j).sum()/(m|j).sum(),4))
