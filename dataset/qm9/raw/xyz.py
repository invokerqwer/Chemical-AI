import numpy as np
import time
from tqdm import tqdm, trange
data=np.load('qm9_eV.npz')
ele={1:'H',6:'C',7:'N',8:'O',9:'F'}
R = data['R']
Z = data['Z']
N= data['N']
split = np.cumsum(N)
R_qm9 = np.split(R, split)
Z_qm9 = np.split(Z,split)
for i in tqdm(range(130832)):
    f=open('xyz/'+str(i)+'.xyz','w')
    print(len(Z_qm9[i]),file=f)
    print('charge=0',file=f)
    for j in range(len(Z_qm9[i])):
        print(ele[Z_qm9[i][j]],R_qm9[i][j][0],R_qm9[i][j][1],R_qm9[i][j][2],file=f)