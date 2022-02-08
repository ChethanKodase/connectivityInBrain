import numpy as np
import torch
import torchvision
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# We can go inside this class and write some methods

class ConnectivityCheck:
    #pass #temporarily we will not recieve any errors inside this class . Now a class is created 
    #methods are functions that are inside the classes using def. If they are created outside the classes you call it definitions
    
    def __init__(self, init_pt_name : str,  destin_pt_name : str, edge_collection ):
        # run validations two the recieved arguments

        assert edge_collection.shape[0] > 0., f"insuficient number of edges in the collection" 
        
        # assign to self object

        #print('edge_collection.shape[0]', edge_collection.shape[0])
        
        self.init_pt_name = init_pt_name
        self.distin_pt_name = destin_pt_name
        self.edge_collection = edge_collection
        #self.price = price
        #self.quantity = quantity


        # unique print line where I can say where is the print line coming from


    # function to get unbranched edges from the other side till there is branch

    def other_side_pts_before_branching(self, actual_new_test, my_edge):
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



    def find_connections(self, edges_collection):

        return edges_collection

    def calculate_total_price(self):
        #python passes the object itself as a first argument when you go aheada and call those methods

        #here we don't have to send those parameters because we assign those attributes once the instancces have been creagted
        return self.price * self.quantity


#now to create some instances of this class

current_edge_collec = torch.tensor([[10., 11.],
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

#current_edge_collec = torch.tensor([])

item1 = ConnectivityCheck("sample_pt_init_1", "sample_pt_dest_2", current_edge_collec) #for each instance that is created like this , it will go and call the double underscore init method
#  This is because the python in the backgroind passes this instance itself as the first argument. 
# random_str = str("4") # is equivalent to "4" .... this was just for example 
#item1.price = 100
#item1.quantity = 5
# above is how you assign attributes to instances

#print(item1.calculate_total_price(item1.price, item1.quantity) )
# when you call a method from an instance
# In 'item1.calculate_total_price(item1.price, item1.quantity)' item1 in the beginneing is passed as first argument 

item2 = ConnectivityCheck("sample_pt_init_3", "sample_pt_dest_4", current_edge_collec)
#item2.price = 1000
#item2.quantity = 3
#print( "item1.calculate_total_price() ",item1.calculate_total_price() )

#print( "item2.calculate_total_price() ",item2.calculate_total_price() )



print(item1.init_pt_name)
print(item2.init_pt_name)

#Now let us understand how we can create some methods and execute them on our instances

#random_str = "aaa"
#   print(random_str.upper()) 

item2.has_numpads = False


# The fact that you use some attributes in the constructor doesn't mean that you don't add 
# some more that you would like after you instantiate the instances that you would like to 



print('item1.find_connections(current_edge_collec)', item2.find_connections(current_edge_collec))
complex_tri_edge = torch.tensor([[ 5, 7]])
print('item2.right_side_pts_before_branching(current_edge_collec, complex_tri_edge)', item2.other_side_pts_before_branching(current_edge_collec, complex_tri_edge))