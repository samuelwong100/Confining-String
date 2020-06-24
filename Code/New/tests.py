# -*- coding: utf-8 -*-
"""
File Name: tests.py
Purpose: 
Author: Samuel Wong
"""
import numpy as np
from source import Dipole_Full_Grid
from source import within_epsilon

def test_dipole_full_grid():
    DFG = Dipole_Full_Grid(11,7,3)
    hg = DFG.half_grid
    DFG2 = Dipole_Full_Grid(101,301,1)
    hg2 = DFG2.half_grid
    
    assert DFG.num_z == 11
    assert DFG.num_z_half == 5
    assert DFG.num_y == 7
    assert within_epsilon(DFG.z0,-0.5)
    assert within_epsilon(DFG.zf,0.5)
    assert within_epsilon(DFG.y0,-0.3)
    assert within_epsilon(DFG.yf,0.3)
    assert within_epsilon(DFG.vertical_length,0.6)
    assert DFG.num_R == 3
    assert DFG.num_R_interval == 2
    assert DFG.R == 0.2
    assert DFG.left_charge_axis_number == 4
    assert DFG.right_charge_axis_number == 6
    assert within_epsilon(DFG.left_charge_z_position,-0.1)
    assert within_epsilon(DFG.right_charge_z_position,0.1)
    assert within_epsilon(DFG.zy_number_to_position(3,5),(-0.2,0.2))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.05,0.01),(0,0))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.06,0.01),(0.1,0))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.34,0.18),(0.3,0.2))
    
    assert hg.parent_grid is DFG
    assert hg.parent_grid.half_grid is hg
    assert hg.num_z == 6
    assert hg.z0 == -0.5
    assert hg.zf == 0
    assert hg.num_y == 7
    assert hg.num_R_half == 2
    assert hg.num_R_half_interval == 1
    assert within_epsilon(hg.horizontal_length,0.5)
    assert within_epsilon(DFG.get_nearest_position_on_grid(-0.34,0.18),(-0.3,0.2))
    assert hg.left_charge_axis_number == 4
    assert within_epsilon(hg.left_charge_z_position,-0.1)
    
    x=DFG2.create_vector_field(10)
    x[:,:,0] = 1
    x[:,:,-1] = 1
    y=hg2.create_vector_field(10)
    y[:,:,0] = 1
    x_new = hg2.reflect_vector_field(y)
    assert within_epsilon(np.max(x-x_new),0)
    
test_dipole_full_grid()