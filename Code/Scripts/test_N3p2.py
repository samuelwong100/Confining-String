# -*- coding: utf-8 -*-
import sys
sys.path.append("../Tension")
from compute_tension import compute_tension
import time

print('hi this is N3p2')
start = time.time()
compute_tension(N=3,p=2)
end = time.time()
print(end - start)
