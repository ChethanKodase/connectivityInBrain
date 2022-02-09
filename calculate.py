import imp
import numpy as np

import torch
import torchvision
import random
import sympy as sp

import minterpy as mp
from minterpy.pointcloud_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import ConnectivityCheck
from ConnectivityCheck import other_side_pts_before_branching, find_connections, _compute_distance_matrix, __repr__, calculate_total_price

#now to create some instances of this class

edge_collection_1 = torch.tensor([[10., 11.],
        [4., 15.],
        [14., 15.],
        [11., 12.],
        [2., 3.],
        [ 5., 14.],
        [11., 9.],
        [ 12., 13.],
        [7., 8.],
        [ 3., 4.],
        [9., 2.],
        [13., 5.],
        [8., 4.],
        [ 1.,  2.],
        [1., 5.]])

#edge_collection_1 = torch.tensor([])

item1 = ConnectivityCheck("sample_pt_init_1", "sample_pt_dest_2", edge_collection_1, 5, 7) 

edge_collection_2 = torch.tensor([[30., 5.],
        [1., 30.],
        [1., 14.],
        [14., 2.],
        [2., 31.],
        [31., 1.],
        [2., 13.],
        [2., 3.],
        [8., 4.],
        [3., 4.],
        [13., 18.],
        [18., 11.],
        [11., 9.],
        [4., 9.],
        [4., 17.],
        [11., 15.],
        [17., 19.],
        [28., 8.],
        [28., 29.],
        [7., 29.],
        [19., 16.],
        [15., 16.],
        [13., 32.],
        [32., 33.],
        [33., 34.],
        [34., 35.],
        [35., 16.]])

item2 = ConnectivityCheck("sample_pt_init_3", "sample_pt_dest_4", edge_collection_2, 5, 7)


edge_collection_3 = torch.tensor([[1., 2.],
        [3., 4.],
        [3., 5.],
        [1., 5.],
        [1., 4.],
        [2., 3.]])

other_side_unbranched_pts_1 = item2.other_side_pts_before_branching(edge_collection_1, 5, 7)
print('item2.find_connections(edge_collection_1,other_side_unbranched_pts, 5, 7)' ,item2.find_connections(edge_collection_1,other_side_unbranched_pts_1, 5, 7))


other_side_unbranched_pts_2 = item2.other_side_pts_before_branching(edge_collection_2, 5, 7)
print('item2.find_connections(edge_collection_1,other_side_unbranched_pts, 5, 7)' ,item2.find_connections(edge_collection_2,other_side_unbranched_pts_2, 5, 7))


other_side_unbranched_pts_3 = item2.other_side_pts_before_branching(edge_collection_3, 1, 3)
print('item2.find_connections(edge_collection_1,other_side_unbranched_pts, 5, 7)' ,item2.find_connections(edge_collection_3,other_side_unbranched_pts_3, 1, 3))


#Accessing the class attribute
print('ConnectivityCheck.num_points' ,ConnectivityCheck.num_points)


#overriding the number of points ie num_points at instance level


# to get minterpy points on sphere
x, y, z = sp.symbols('x y z')

expr = x**2 + y**2 + z**2 - 1
poly = sp.Poly(expr, x, y, z)

# convert sympy polynomial to minyterpy polynomial
newt_poly = sympy_to_mp(poly, mp.NewtonPolynomial)


#sample points
#Accessing the class attribute num_points
point_data = sample_points_on_poly(ConnectivityCheck.num_points,        # Number of points to be sampled
                                   newt_poly,  # Polynomial in Newton basis
                                   bounds=1, # Boundary of the Cubic domain to be sampled
                                   tol=1e-15)  # Tolerance in solution

point_data = torch.tensor(point_data)
print('point_data.shape', point_data.shape)

# we can also access those attributes using those instance labels like 
# item1.num_points or item2.num_points

# to see all the attributes belonging to that specific object
#print('ConnectivityCheck.__dict__',ConnectivityCheck.__dict__)

#print('item1.__dict__',item1.__dict__)

# now compute distane matrix

print( 'item1._compute_distance_matrix(point_data, p=2)',item1._compute_distance_matrix(point_data, p=2).shape)

print('ConnectivityCheck.all',ConnectivityCheck.all) # this gives a list of five instances

#these are uselful while pasting them to python console
for instance in ConnectivityCheck.all:
    print('instance.init_pt_name',instance.init_pt_name)
    print('instance.destin_pt_name',instance.destin_pt_name)