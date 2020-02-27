# -*- coding: utf-8 -*-

import sys
sys.path.append("../Tensions")
from Tensions import compute_tension
import time

start = time.time()
compute_tension(N=7,p=2)
end = time.time()
print(end - start)
