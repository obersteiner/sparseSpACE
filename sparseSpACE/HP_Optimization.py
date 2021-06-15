import sparseSpACE.DEMachineLearning as deml
import numpy as np
import scipy as sp
import math
import random
from scipy.optimize import fmin

#performs Grid Optimization
def perform_GO_classification(data):
	#sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15)
	#data = deml.DataSet(sklearn_dataset, name='Input_Set')
	#for storing the current best evaluation and time
	best_evaluation = None
	best_time = None
	#storing the parameters of the best evaluation
	best_lambd = None
	best_masslump = None
	best_min_lv = None
	best_one_vs_others = None
	#For Testing best evaluation: best_evaluation = 0.7
	#print("original best evaluation: " + str(best_evaluation))
	#todo: split Dataset
	#original classification: classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
	#original settings: classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True)
	#perform grid search for a few values of masslumping and lambd
	lambd_levels = 10
	#Note: vllt besser logarithmisch, sehr kleine Werte oft besser (1/10, 1/100, 1/1000)
	for cur_lambd in range (0, lambd_levels+1, 5):
		#Note: lamd vor Allem sinnvoll wenn masslumping aus ist, sollte kaum unterschied machen wenn
		for cur_massl in [True, False]:
			for cur_min_lv in range (1, 4, 1):
				for cur_one_vs_others in [True, False]:
					classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
					print("current masslumping = " + str(cur_massl) + "; current lambd = " + str(cur_lambd/lambd_levels) + "; current min lv = " + str(cur_min_lv) + "; current one vs others = " + str(cur_one_vs_others))
					classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd/lambd_levels, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
					cur_time = classification._time_used
					print ("current time needed = " + str(cur_time))
					evaluation = classification.evaluate()
					print("Percentage of correct mappings",evaluation["Percentage correct"])
					#currently only looking at Percentage correct and time needed for evaluating
					cur_evaluation = evaluation["Percentage correct"]
					if best_evaluation == None or cur_evaluation > best_evaluation or (cur_evaluation == best_evaluation and (best_time == None or best_time>cur_time)):
						best_evaluation = cur_evaluation
						best_lambd = cur_lambd
						best_time = cur_time
						best_masslump = cur_massl
						best_min_lv = cur_min_lv
						best_one_vs_others = cur_one_vs_others
						print("Best evaluation is now " + str(best_evaluation) + " with masslump = " + str(best_masslump) + ", lambd = " + str(best_lambd) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others) + " and time = " + str(best_time))
					else:	print("Best evaluation is still " + str(best_evaluation) + " with masslump = " + str(best_masslump) + ", lambd = " + str(best_lambd) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others) + " and time = " + str(best_time))
					print ("END OF CURRENT EVALUATION \n")
	print("In the end, best evaluation is " + str(best_evaluation)  + " with masslump = " + str (best_masslump) + ", lamb = " + str(best_lambd) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others) + " and time = " + str(best_time))

#returns evaluation for certain Parameters on a certain data set
def perform_evaluation_at(cur_data, cur_lambd: float, cur_massl: bool, cur_min_lv: int, cur_one_vs_others: bool):
	classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
	classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
	evaluation = classification.evaluate()
	print("Percentage of correct mappings",evaluation["Percentage correct"])
	return evaluation["Percentage correct"]

#TODO: will perform Bayesian Optimization; Inputs are the data, amount of iterations(amt_it)
def perform_BO_classification(data, amt_it: int, dim: int):
	#notes how many x values currently are in the evidence set - starts with amt_HP+1
	amt_HP = 4
	cur_amt_x: int = amt_HP+1
	print("I should do Bayesian Optimization for " + str(amt_it) + " iterations!")
	C = create_evidence_set(data, amt_it, amt_HP)
	C_x=C[0]
	C_y=C[1]
	print(C)
	K_matr = get_cov_matrix(C_x, cur_amt_x)
	print(K_matr)
	#dummy value for beta
	beta = 0.5
	mu = lambda x: get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(C_y))
	sigma_sqrd = lambda x: k(x, x)-get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(get_cov_vector(C_x, x, cur_amt_x)))
	sigma = lambda x: math.sqrt(sigma_sqrd(x))
	alpha = lambda x: mu(x)+(math.sqrt(beta))*sigma(x)
	#negates alpha bc maximum has to be found
	alpha_neg = lambda x: -alpha(x)
	stupid_value=np.array([1, 1, 1, 1])
	for i in range (0, amt_it, 1):
		print("iteration: " + str(i))
		beta = get_beta(i)
		print("beta: " + str(beta) + " mu(x): " + str(mu(stupid_value)) + " sigma(x): " + str(sigma(stupid_value)))
		print(alpha(stupid_value))
		#value that will be evaluated and added to the evidence set
		print(fmin(alpha_neg, [0, 0, 0, 0]))
		#new_x=fmin(alpha_neg, [0, 0, 0, 0])
		#print("current new_x: " + str(new_x) + " with function value: " + str(alpha(new_x)))
		cur_amt_x += 1
		

#use squared exp. kernel as covariance function k, with parameters sigma_k and l_k
sigma_k = 1
l_k = 0.5
#since creation of covariance matrices is done in another function k(x) needs to be outside
k = lambda x, y: (sigma_k**2)*math.exp((-0.5/l_k**2)*np.linalg.norm(x-y)**2)

#cur_length is needed so that rest of matrix stays Id
def get_cov_matrix(x, cur_length: int):
	K = np.identity(len(x))
	for i in range (0, cur_length, 1):
		for j in range (0, cur_length, 1):
			K[i][j]=k(x[i], x[j])
	return K

#returns the vector k for a certain input new_x
def get_cov_vector(ev_x, new_x, cur_length: int):
        #somehow the rest of the function would still work even if x is wrong size, but no idea why / what it does. Better to check
        if(len(ev_x[0]) != len(new_x)):
                print("Input x has the wrong size")
                return None
        k_vec = np.zeros(len(ev_x))
        for i in range (0, cur_length, 1):
                k_vec[i] = k(ev_x[i], new_x)
        return k_vec

def create_evidence_set(data, amt_it: int, dim_HP: int, is_random: bool = True):
	x = np.zeros(shape=(amt_it+dim_HP+1, dim_HP))
	y = np.zeros((amt_it+dim_HP+1))
	print(x)
	if(is_random):
		for i in range(0, dim_HP+1, 1):
			new_x = create_random_x()
			while(check_if_in_array(new_x, x)):
				new_x = create_random_x()
			x[i] = new_x
			#evaluating takes quite some time, especially for min_lv>1
			y[i] = 0.97 #hard-coded y for faster testing of the rest
#perform_evaluation_at(data, new_x[0], new_x[1], new_x[2], new_x[3])
	#One could hard-code non-random values for testing
	return x, y

#returns a random x for the given purpose
def create_random_x(purpose = "classification"):
	if(purpose == "classification"):
		return [random.random(), random.randint(0, 1), random.randint(1, 3), random.randint(0, 1)]
	else:
		print("Invalid Input")

#checks if array has the element x, returns True if yes. Caution: If x has same length but different dimension to the elements of array this may give unintended results
#maybe later a threshhold can be added if only part of the matrix should be checked
def check_if_in_array(x, array):
    if(len(x) != len(array[0])):
        #print("not same size")
        return False
    for i in range(0, len(array), 1):
        if((x == array[i]).all()):
            return True
            break
    return False

#TODO: will calculate and return current beta
def get_beta(cur_it: int):
	beta = cur_it/2
	return beta

#Created newer, better method above, but this may be helpful for the hardcoding stuff
def get_init_evidence(data, dim: int, amt_it: int):
	#hard-coded x-values. Order:[lambd: float, massl:bool, min_lv: 0<int<5, one_vs_others: bool]
	#But: all values as float, rounded as neccessary. Bool: 0=False
	#TODO: x-values should eventually be random, and 1+dim(=amt of Parameters) many
	x = np.array([[0.1, 1, 1, 1], [1.1, 0, 3, 1], [0.005, 1, 2, 0], [2.0, 0, 1, 0], [0.00001, 1, 1, 1]])
	#y-values should be evaluation at x
	y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
	return x, y


#Basically copied code from Tutorial_DEMachineLearning
#prepare Dataset
samples = 500
dimension = 2
labels = 6
#vllt anderes dataset ausprobieren?, vllt höhere dimensionen, bis zu dim=10. Note: moons immer 2d, aber z.B.
#gaussian quantiles, classification.. kann man einstellen
sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15, random_state=1)
data1 = deml.DataSet(sklearn_dataset, name='Input_Set')
data_moons = data1.copy()
data_moons.set_name('Moon_Set')
perform_BO_classification(data_moons, 6, dimension)
