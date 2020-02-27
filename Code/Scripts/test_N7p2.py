# -*- coding: utf-8 -*-
import sys
sys.path.append("../Tension")
from compute_tension import compute_tension
import time

print('hi this is N7p2')
start = time.time()
compute_tension(N=7,p=2)
end = time.time()
print(end - start)
