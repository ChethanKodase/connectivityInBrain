import numpy as np

import torch
import torchvision
import random
import sympy as sp

import minterpy as mp
from minterpy.pointcloud_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# We can go inside this class and write some methods

class ConnectivityCheck:

    #writing a class attribute
    num_points = 100

    all = [] # this is to add our instances to this list each time we are going to create a new instance
    #pass #temporarily we will not recieve any errors inside this class . Now a class is created 
    #methods are functions that are inside the classes using def. If they are created outside the classes you call it definitions
    

    #the below method is called immediately once the instance is created
    def __init__(self, init_pt_name : str,  destin_pt_name : str, edge_collection, init_pt, dest_pt):
        # run validations two the recieved arguments

        assert edge_collection.shape[0] > 0., f"insuficient number of edges in the collection" 
        
        # assign to self object

        #print('edge_collection.shape[0]', edge_collection.shape[0])
        
        self.init_pt_name = init_pt_name
        self.destin_pt_name = destin_pt_name
        self.edge_collection = edge_collection
        self.init_pt = init_pt
        self.dest_pt = dest_pt

        ConnectivityCheck.all.append(self)

        # unique print line where I can say where is the print line coming from


    # function to get unbranched edges from the other side till there is branch

    def other_side_pts_before_branching(self, actual_new_test, init_pt, dest_pt):

        my_edge = torch.tensor([[init_pt, dest_pt]])
        #print('edge_collection.shape[0]', actual_new_test.shape[0])
        left_ind = my_edge[0][1]
        right_ind = my_edge[0][0]
        found_right_ind = False
        going_nowhere= False

        new_test = actual_new_test
        actual_new_test_an = actual_new_test
        
        tracker = 0
        no_branches_formed = True
        loop_tracker = 0
        positions1 = (new_test == left_ind).nonzero(as_tuple=False)
        loops_collec = []
        current_loop = torch.tensor([])
        consec_pt_tracker = torch.tensor([])
        while (not(found_right_ind) or not(going_nowhere)):

            positions1 = (new_test == left_ind).nonzero(as_tuple=False)
            
            if(positions1.shape[0]>1):
                break

            branches_rising = positions1.shape[0]

            if(positions1.shape[0]==0):
                #lets see
                break

            else:
                first_position = positions1[0][0]

                adj_edge1 = new_test[positions1[0][0]]
                other_end1 = abs(positions1 - torch.tensor([[0, 1]]))


                consec_pt1 = new_test[other_end1[0][0]][other_end1[0][1]]
                consec_pt1s = torch.unsqueeze(consec_pt1,0)
                consec_pt_tracker = torch.cat((consec_pt_tracker, consec_pt1s),0)
                consec_pt1 = int(consec_pt1)

                current_loop = torch.cat((current_loop,adj_edge1),0)
                current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)
                
                if(consec_pt1 == my_edge[0][0]):
                    current_loop = torch.tensor([])
                    
                if(consec_pt1 == right_ind):
                    my_edge1 = torch.squeeze(my_edge,0)
                    current_loop = torch.cat((current_loop,my_edge1),0)
                    current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)                

                    loops_collec.append(current_loop1)

                    loop_tracker = loop_tracker + 1
                left_ind = consec_pt1
                new_test = torch.cat((new_test[:first_position], new_test[first_position+1:]))
                tracker = tracker+1
        
        return consec_pt_tracker

    # function to check whether the selected edge is going to close a potential loop

    def find_connections(self, actual_new_test,other_side_unbranched_pts, init_pt, dest_pt):

        my_edge = torch.tensor([[init_pt, dest_pt]])
        #other_side_unbranched_pts = other_side_pts_before_branching(actual_new_test, my_edge)
        #print('other_side_unbranched_pts.shape',other_side_unbranched_pts.shape[0])
        left_ind = my_edge[0][0]
        right_ind = my_edge[0][1]
        found_right_ind = False
        going_nowhere= False

        new_test = actual_new_test
        actual_new_test_an = actual_new_test
        
        tracker = 0
        no_branches_formed = True
        loop_tracker = 0
        positions1 = (new_test == left_ind).nonzero(as_tuple=False)
        loops_collec = []
        current_loop = torch.tensor([])
        consec_pt_tracker = torch.tensor([])
        while (not(found_right_ind) or not(going_nowhere)):

            positions1 = (new_test == left_ind).nonzero(as_tuple=False)
            #print(new_test)
            #print()
            #print('positions1.shape[0]',positions1.shape[0])
            #print()
            
            if(positions1.shape[0]>1):
                #edg_q_del = new_test[positions1[0][0]]
                other_end_con = abs(positions1 - torch.tensor([[0, 1]]))
                consec_pt_con = new_test[other_end_con[0][0]][other_end_con[0][1]]
                #print('did i get consec_pt_con ', consec_pt_con)
                #print('now check if it works', not(consec_pt_con in consec_pt_tracker))
                
                if(not(other_side_unbranched_pts.shape[0] == 0)):
                    if(not(consec_pt_con in consec_pt_tracker) and not(consec_pt_con==other_side_unbranched_pts[-2])):
                        edge_to_delete = new_test[positions1[0][0]]
                else:
                    if(not(consec_pt_con in consec_pt_tracker)):
                        edge_to_delete = new_test[positions1[0][0]]                
                no_branches_formed = False
                #print('edge_to_delete first',edge_to_delete)
            branches_rising = positions1.shape[0]

            if(positions1.shape[0]==0):
                current_loop = torch.tensor([])
                consec_pt_tracker = torch.tensor([])
                #going_nowhere= True
                '''if(no_branches_formed):
                    break'''
                
                left_ind = my_edge[0][0]

                deletable_edge_position1 = (actual_new_test == edge_to_delete[0]).nonzero(as_tuple=False)
                deletable_edge_position2 = (actual_new_test == edge_to_delete[1]).nonzero(as_tuple=False)

                deletable_edge_position1 = deletable_edge_position1[:,0]

                deletable_edge_position2 = deletable_edge_position2[:,0]

                a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
                deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]
                #print()
                #print('deletable_row_position',deletable_row_position)
                
                if(deletable_row_position.shape[0]==0):
                    #going_nowhere = True
                    current_loop = torch.tensor([])
                    break

                deletable_row_position = deletable_row_position[0]
                
                #print('Does my edge to delete contain my edge left index ? ', my_edge[0][0] in edge_to_delete)
                #print()
                actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
                if(my_edge[0][0] in edge_to_delete):

                    deletable_edge_position1 = (actual_new_test_an == edge_to_delete[0]).nonzero(as_tuple=False)
                    deletable_edge_position2 = (actual_new_test_an == edge_to_delete[1]).nonzero(as_tuple=False)

                    deletable_edge_position1 = deletable_edge_position1[:,0]

                    deletable_edge_position2 = deletable_edge_position2[:,0]

                    a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
                    deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]
                    #print()
                    #print('deletable_row_position',deletable_row_position)

                    if(deletable_row_position.shape[0]==0):
                        #going_nowhere = True
                        current_loop = torch.tensor([])
                        break

                    deletable_row_position = deletable_row_position[0]
                    
                    actual_new_test_an = torch.cat((actual_new_test_an[:deletable_row_position], actual_new_test_an[deletable_row_position+1:]))    
                    actual_new_test = actual_new_test_an
                    
                #actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
                #print('what is this', actual_new_test)
                new_test = actual_new_test

                positions1 = (new_test == left_ind).nonzero(as_tuple=False)
                #print('whats happening here',positions1.shape )
                #print('is the same edge still to delete', edge_to_delete)
                if(tracker ==0):
                    break

                '''if(positions1.shape[0]>1):
                edge_to_delete = new_test[positions1[0][0]]
                no_branches_formed = False'''
            else:
                first_position = positions1[0][0]
                #print('first_position',first_position)
                adj_edge1 = new_test[positions1[0][0]]
                other_end1 = abs(positions1 - torch.tensor([[0, 1]]))


                consec_pt1 = new_test[other_end1[0][0]][other_end1[0][1]]
                consec_pt1s = torch.unsqueeze(consec_pt1,0)
                consec_pt_tracker = torch.cat((consec_pt_tracker, consec_pt1s),0)
                consec_pt1 = int(consec_pt1)

                    
                #print('consec_pt1',consec_pt1)
                #print('adj_edge1',adj_edge1)
                current_loop = torch.cat((current_loop,adj_edge1),0)
                current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)
                #print('consec_pt_tracker',consec_pt_tracker)
                
                if(consec_pt1 == my_edge[0][0]):
                    current_loop = torch.tensor([])
                    
                if(consec_pt1 == right_ind):
                    my_edge1 = torch.squeeze(my_edge,0)
                    current_loop = torch.cat((current_loop,my_edge1),0)
                    current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)                
                    #found_right_ind = True
                    #print('current_loop',current_loop1)
                    #current_loop1 = torch.unsqueeze(current_loop1,0)
                    #print('current_loop shape now',current_loop1.shape)
                    #print('loop_tracker', loop_tracker)
                    loops_collec.append(current_loop1)
                    #loops_collec[loop_tracker] = current_loop1
                    loop_tracker = loop_tracker + 1
                    #print()
                    #print("Wow! Found a loop here")
                    #print()
                    #break

                #else:

                left_ind = consec_pt1
                new_test = torch.cat((new_test[:first_position], new_test[first_position+1:]))
                #print('new_test',new_test)
                tracker = tracker+1
        
        #loops_collec = torch.FloatTensor(loops_collec)
        return loops_collec


    def _compute_distance_matrix(self, x, p=2):
        x_flat = x.view(x.size(0), -1)

        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

        return distances

    def __repr__(self):
        return f"ConnectivityCheck('{self.init_pt_name}','{self.destin_pt_name}','{self.init_pt}','{self.dest_pt}')"


    '''
    def find_connections(self, edges_collection):

        return edges_collection'''

    def calculate_total_price(self):
        #python passes the object itself as a first argument when you go aheada and call those methods

        #here we don't have to send those parameters because we assign those attributes once the instancces have been creagted
        return self.price * self.quantity

    # in static method we never send the object as the first argument
    # This should do something that has relationship with the class, but not something that must be unique per instance
    @staticmethod
    def is_integer(num):



        ''' Class method should also do something that has a relationship with the class, but usualy, those are used to manipulate 
        different structures of data to instantiate objetcs, like we have done with csv '''

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

