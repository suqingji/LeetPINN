import glob, subprocess, os, sys
envs = sorted(glob.glob('/home/sqj/miniconda3/envs/*/'))
bins = ['/home/sqj/miniconda3/bin/python'] + [e+'/bin/python' for e in envs]
for b in bins:
    label = b
    try:
        out = subprocess.check_output([b, '-c', 'import torch,matplotlib;print(torch.__version__, torch.cuda.is_available(), matplotlib.__version__)'], stderr=subprocess.STDOUT, text=True, timeout=20)
        print(label, '->', out.strip())
    except Exception as ex:
        print(label, '-> ERR', str(ex)[:80])
