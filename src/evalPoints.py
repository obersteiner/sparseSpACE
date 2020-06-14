#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpi4py import MPI 


def start_parallel_evaluation(f,new_points):     
        len_points=len(new_points)        
        if len_points>0:
                comm=MPI.COMM_WORLD
                rank=comm.Get_rank()
                size=comm.Get_size()
                data=[]
                
                if (rank==0):
                    
                    step=len_points // size
                    remainder=len_points % size
                    counter=len_points % size
                    for i in range(size):
                        if i==0:                                                        
                            if remainder>0:
                                           data=new_points[:(step+1)]                                           
                                           counter-=1 
                            else: 
                                data=new_points[:step]
                          
                        else:
                            
                            part_to_eval=step*(i)
                            if counter>0:                                                               
                                part_to_eval+=remainder-counter
                                comm.send(new_points[part_to_eval:(part_to_eval+step+1)],dest=i)                                           
                                counter-=1                                            
                            else:                                
                                if size<len_points:
                                    part_to_eval+=remainder 
                                    comm.send(new_points[part_to_eval:(part_to_eval+step)],dest=i)
                                                       
                                else:
                                          comm.send([],dest=i)
                       
                if (len(data))>0:
                        for i in range(len(data)):
                                  f(data[i])
                barrier=False                
                for i in range(size-1):    
                    barrier=comm.recv(source=i+1)  
                

