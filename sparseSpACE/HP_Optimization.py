import sparseSpACE.DEMachineLearning as deml
import numpy as np
import scipy as sp
import math
import random
from scipy.optimize import fmin
from scipy.optimize import minimize

#class HP_Optimization gets (data not needed bc only pea needs that which is the function!),
#the function on which HP should be Optimized and search space for the HPs
#space is in form [["interval", <start>, <end>], ["list", <l0>, <l1>, <l2>,...], ....]
class HP_Optimization:
	def __init__(self, function, hp_space):
		self.function = function
		self.hp_space = hp_space

	#performs Grid Optimization
	def perform_GO_classification(self, lambd_levels):
		#sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15)
		#data = deml.DataSet(sklearn_dataset, name='Input_Set')
		#for storing the current best evaluation and time
		best_evaluation = None
		#best_time = None
		#storing the parameters of the best evaluation
		best_lambd = None
		best_masslump = None
		best_min_lv = None
		best_one_vs_others = None
		#todo: split Dataset??
		#original classification: classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
		#original settings: classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True)
		#perform grid search for lambd_levels many lambd between 0 and 1 and all values for the rest
		#Note: lambd vllt besser logarithmisch, sehr kleine Werte oft besser (1/10, 1/100, 1/1000)
		#für for-schleife karthesisches produkt aus HO um alle tupel zu bilden und über diese tupel zu iterieren für beliebig dimensional
		for cur_lambd_it in range (0, lambd_levels, 1):
			cur_lambd = cur_lambd_it/lambd_levels
			#Note: lambd vor Allem sinnvoll wenn masslumping aus ist, sollte kaum unterschied machen wenn an
			for cur_massl in [True, False]:
				for cur_min_lv in range (1, 3, 1):
					for cur_one_vs_others in [True, False]:
						#use "perform evaluation at"
						#cur_time = classification._time_used
						#print ("current time needed = " + str(cur_time))
						cur_evaluation = perform_evaluation_at(cur_lambd, cur_massl, cur_min_lv, cur_one_vs_others)
						print("Percentage of correct mappings",cur_evaluation)
						#if best_evaluation == None or cur_evaluation > best_evaluation or (cur_evaluation == best_evaluation and (best_time == None or best_time>cur_time)):
						if(best_evaluation == None or cur_evaluation>best_evaluation):
							best_evaluation = cur_evaluation
							best_lambd = cur_lambd
							#best_time = cur_time0
							best_masslump = cur_massl
							best_min_lv = cur_min_lv
							best_one_vs_others = cur_one_vs_others
							print("Best evaluation is now " + str(best_evaluation) + " with lambd = " + str(best_lambd) + ", masslump = " + str(best_masslump) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others)) #+ " and time = " + str(best_time))
						else:	print("Best evaluation is still " + str(best_evaluation) + " with lambd = " + str(best_lambd) + ", masslump = " + str(best_masslump) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others)) #+ " and time = " + str(best_time))
						print ("END OF CURRENT EVALUATION \n")
		print("In the end, best evaluation is " + str(best_evaluation)  + " with lambd = " + str(best_lambd) + ", masslump = " + str(best_masslump) + ", min_level = " + str(best_min_lv) + ", one_vs_others = " + str(best_one_vs_others))
		#+ " and time = " + str(best_time))

	#performs random optimization
	def perform_RO_classification(self, amt_it: int, dim: int):
		best_evaluation = 0
		best_x = [1, 1, 1, 1]
		for i in range (0, amt_it, 1):
			x = create_random_x()
			new_eval = perform_evaluation_at(x[0], x[1], x[2], x[3])
			if new_eval>best_evaluation:
				best_x = x
				best_evaluation = new_eval
		print("Best evaluation in " + str(amt_it) + " random steps: " + str(best_evaluation) + " at " + str(x))

	#returns evaluation for certain Parameters on a certain data set
	def perform_evaluation_at(self, cur_lambd: float, cur_massl: bool, cur_min_lv: int, cur_one_vs_others: bool):
		#classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
		#classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
		#evaluation = classification.evaluate()
		#print("Percentage of correct mappings",evaluation["Percentage correct"])
		##wenn Zeit mit reingerechnet werden soll o.ä. in dieser Funktion
		#return evaluation["Percentage correct"]
		return self.function(cur_lambd, cur_massl, cur_min_lv, cur_one_vs_others)

	classification_space = [["interval", 0, 1], ["list", 0, 1], ["list", 1, 2, 3], ["list", 0, 1]]

	#TODO: will perform Bayesian Optimization; Currently basically performs BO with Naive rounding and a weird beta
	#Note: HP Input of classification is (lambd:float, massl:bool, min_lv:int <4, one_vs_others:bool)
	def perform_BO_classification(self, amt_it: int, function = perform_evaluation_at, space = classification_space, ev_is_rand: bool = True):
		#notes how many x values currently are in the evidence set - starts with amt_HP+1
		amt_HP = len(space)
		cur_amt_x: int = amt_HP+1
		x_ret = None
		y_ret = None
		print("I should do Bayesian Optimization for " + str(amt_it) + " iterations, ev_is_rand is " + str(ev_is_rand) + " !")
		C = create_evidence_set(amt_it, amt_HP, ev_is_rand)
		C_x=C[0]
		print("C_x: " + str(C_x))
		C_y=C[1]
		print("C_y: " + str(C_y))
		#print("Evidence Set: \n" + str(C))
		for i in range (0, amt_it, 1):
			print("iteration: " + str(i))
			beta = get_beta(i+1)
			l = get_l_k(i)
			if(np.linalg.det(get_cov_matrix(C_x, cur_amt_x))==0):
				print("With the nex x value added the Covariance Matrix is singular.\nBayesian Optimization can not be continued and will be ended here, at the start of iteration " + str(i))
				break
			if((get_cov_matrix(C_x, cur_amt_x) == np.identity(len(C_x))*(cur_amt_x+10)).all()):
				print("cov_matr has been modified to be Id*scalar! This means it was singular -> ending at iteration " + str(i))
				break
			else:
				print("Apparently the cov_matr is not singular at the beginning of iteration " + str(i))

			#value that will be evaluated and added to the evidence set
			print("Getting new x:")
			new_x=acq_x(beta, l, C_x, C_y, cur_amt_x)
			print("new x: " + str(new_x))
			new_x_rd = round_x_classification(new_x)
			print("new x rd: " + str(new_x_rd))
			#erstelle neues beta und l und berechne neuen wert damit
			while(check_if_in_array(new_x_rd, C_x)):
				print("!!!!!!!!!!!Need new beta and l")
				old_x_rd = new_x_rd
				print("old x rd was: " + str(old_x_rd))
				beta_and_l = get_new_beta_and_l(beta, cur_amt_x, new_x, C_x, C_y)
				new_x = acq_x(beta_and_l[0], beta_and_l[1], C_x, C_y, cur_amt_x)
				new_x_rd = round_x_classification(new_x)
				print("new x rd is: " + str(new_x_rd))
				if(old_x_rd==new_x_rd):
					print("We're in an infinite loop! Getting out")
					#problem: after ending infinite loop the value is added, even though it was already in C_x
					#..this makes the matrix singular. Shouldn't happen though with modified get_beta_and_l
					break
			print("out of loop")
			if(old_x_rd==new_x_rd):
				print("There was an infinite loop when getting new x. Ending BO at iteration " + str(i))
				break
			print("adding " + str(new_x_rd))
			C_x[cur_amt_x] = new_x_rd
			C_y[cur_amt_x] = perform_evaluation_at(new_x_rd[0], new_x_rd[1], new_x_rd[2], new_x_rd[3])
			cur_amt_x += 1
			#ends everything if cov_matr is singular. Should this be dependant on l? Cov_matr being singular should not depend on l I think
			print("Checking if cov_matr is singular")
			if(np.linalg.det(get_cov_matrix(C_x, cur_amt_x)) == 0):
				print("With the nex x value added the Covariance Matrix is singular.\nBayesian Optimization can not be continued and will be ended here, after " + str(i) + " iterations")
				break
			if((get_cov_matrix(C_x, cur_amt_x) == np.identity(len(C_x))*(cur_amt_x+10)).all()):
				print("cov_matr has been modified to be Id*scalar! This means it was singular -> ending at iteration " + str(i))
				break
			else:
				print("Apparently the cov_matr is not singular after iteration " + str(i))

		for i in range(0, cur_amt_x, 1):
			if(y_ret == None or y_ret<C_y[i]):
				y_ret = C_y[i]
				x_ret = C_x[i]
		print("The Best value found in " + str(amt_it) + " iterations is " + str(y_ret) + " at " + str(x_ret))
		return x_ret, y_ret
	#use squared exp. kernel as covariance function k, with parameters sigma_k and l_k
	sigma_k = 1
	l_k = 0.5
	#since creation of covariance matrices is done in another function k(x) needs to be outside
	k = lambda x, y: (sigma_k**2)*math.exp((-0.5/l_k**2)*np.linalg.norm(x-y)**2)

	#cur_length is needed so that rest of matrix stays Id
	def get_cov_matrix(self, C_x, cur_length: int):
		K = np.identity(len(C_x))
		for i in range (0, cur_length, 1):
			for j in range (0, cur_length, 1):
				K[i][j]=k(C_x[i], C_x[j])
		if(np.linalg.det(K) == 0):
			print("Oh no! The covariance matrix is singular!! It is " + str(K))
			print("Current C_x is: " + str(C_x))
			#Maybe change matrix if it is singular? E.g. to scalar*Id, so it's def. not the smallest?
			#Would have to deal with that in other methods though
			#Important!! K is inverted. But also it's used some way in g?? Unsure what to do
			#at least it doesn't crash rn
			#add small values to diagonal? -> verfälscht sachen. Müsste relativ zu werten in Matrix angepasst werden
			#z.b. 10^-15 oder so
			K = np.identity(len(C_x))*(cur_length+10)
		return K

	#returns the vector k for a certain input new_x
	def get_cov_vector(self, ev_x, new_x, cur_length: int):
	        #somehow the rest of the function would still work even if x is wrong size, but no idea why / what it does. Better to check
	        if(len(ev_x[0]) != len(new_x)):
	                print("Input x has the wrong size")
	                return None
	        k_vec = np.zeros(len(ev_x))
	        for i in range (0, cur_length, 1):
	                k_vec[i] = k(ev_x[i], new_x)
	        return k_vec

	def create_evidence_set(self, amt_it: int, dim_HP: int, is_random: bool = True):
		x = np.zeros(shape=(amt_it+dim_HP+1, dim_HP))
		y = np.zeros((amt_it+dim_HP+1))
		if(is_random):
			for i in range(0, dim_HP+1, 1):
				new_x = create_random_x()
				while(check_if_in_array(new_x, x)):
					new_x = create_random_x()
				x[i] = new_x
				#evaluating takes quite some time, especially for min_lv>1
				#y[i] = perform_evaluation_at(data, new_x[0], new_x[1], new_x[2], new_x[3])
				#needs to be casted to int bc otherwise it's automatically float which doesn't work
				y[i] = perform_evaluation_at(x[i][0], int(x[i][1]), int(x[i][2]), int(x[i][3]))
		else:
			#hard-code non-random values for testing - current threw error at it. 24
			#use seed for random values instead of hard coded values? -> get used seed
			x[0] = [0.20213575, 0., 1., 0.] #[0.79125794, 0, 1, 0]
			x[1] = [0.80125658, 0., 1., 0.] #[0.69819941, 1, 2, 0]
			x[2] = [0.09898312, 1., 3., 1.] #[0.35823418, 0, 1, 0]
			x[3] = [0.88249225, 1., 1., 0.] #[0.51043662, 1, 1, 0]
			x[4] = [0.1321559, 1., 2., 1.] #[0.54776247, 0, 1, 0]
			for i in range(0, dim_HP+1, 1):
				y[i] = perform_evaluation_at(x[i][0], int(x[i][1]), int(x[i][2]), int(x[i][3]))
		return x, y

	#returns a random x for the given purpose - needs search space
	def create_random_x(self, purpose = "classification"):
		if(purpose == "classification"):
			return [random.random(), random.randint(0, 1), random.randint(1, 3), random.randint(0, 1)]
		else:
			print("Invalid Input")

	#checks if array has the element x, returns True if yes. Caution: If x has same length but different dimension to the elements of array this may give unintended results
	#maybe later a threshhold can be added if only part of the matrix should be checked
	def check_if_in_array(self, x, array):
	    if(len(x) != len(array[0])):
	        #print("not same size")
	        return False
	    for i in range(0, len(array), 1):
	        if((x == array[i]).all()):
	            return True
	            break
	    return False

	#calculates and returns current beta. t is current iteration
	def get_beta(self, t: int):
		r: float = 1.5
		d: int = 4
		delta: float = 0.1
		a: float = 1
		b: float = 1
		beta = 2*math.log(t**2*2*math.pi**2/3*delta)+2*d*math.log(t**2*d*b*r*math.sqrt(math.log(4*d*a/delta)))
		print(beta)
		return beta

	#TODO: get l_k and use it
	def get_l_k(self, t: int):
		return 0.5

	#TODO: get new beta and l. Currently mock function
	def get_new_beta_and_l(self, cur_beta: float, cur_amt_x, cur_x, C_x, C_y):
		global l_k
		print("Getting new beta and l. Current beta: " + str(cur_beta) +", Current l: " + str(l_k))
		if(np.linalg.det(get_cov_matrix(C_x, cur_amt_x))==0):
			print("Getting new beta and l, but suddenly the cov matr is singular??")
		else:
			print("Apparently the cov_matr is not singular. Getting new beta and l")
		#making upper bounds dependable on current values so bounds are never too low
		beta_h = cur_beta+100
		l_h = l_k+50
		#0 if rd(x) is in ev set C_x, constant otherwise (5)
		#penalty p(x) erhöhen wenn gleicher Wert rauskommt. z.B. immer +50 oder *2 oder anders exponentiell
		p = lambda x: check_if_in_array(round_x_classification(x), C_x)*50
		#gets the x value for certain l (z[1]) and beta (z[0])
		new_x = lambda z: acq_x(z[0], z[1], C_x, C_y, cur_amt_x)
		#for g: x[0] is \beta+d\beta, x[1] is l. Also d\beta = \beta+d\beta-\beta
		g = lambda x: (x[0]-cur_beta)+np.linalg.norm(cur_x-new_x(x))+p(new_x(x))
		bounds_g = ((cur_beta, beta_h), (l_k, l_h))
		print("About to minimize g(...)")
		#due to numerical inaccuracies a matrix might become singular with new beta and l
		#(even though it wouldn't be) mathematically - how to deal with that??
		result = minimize(g, [1, 1], method='L-BFGS-B', bounds=bounds_g).x
		print("New beta: " + str(result[0]) + ", New l: " + str(result[1]))
		#result is in the form [new beta, new l]
		return result

	#TODO acquire new x for l, beta, C_x, cur_amt_x, using GP-UCB
	def acq_x(self, beta: float, l: float, C_x, C_y, cur_amt_x):
		global l_k
		old_l=l_k
		l_k=l
		K_matr = get_cov_matrix(C_x, cur_amt_x)
		if(np.linalg.det(K_matr) == 0):
			print("Covariance Matrix is indeed singular!")
			#return a value that's bullshit? like [0, 0, 0, 0] (min lv is >=1)
		#print(K_matr)
		mu = lambda x: get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(C_y))
		sigma_sqrd = lambda x: k(x, x)-get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(get_cov_vector(C_x, x, cur_amt_x)))
		#takes sqrt of abs(sigma_sqrd) bc otherwise fmin gives an error - might be an overall warning sign tho
		sigma = lambda x: math.sqrt(abs(sigma_sqrd(x)))
		alpha = lambda x: mu(x)+(math.sqrt(beta))*sigma(x)
		#negates alpha bc maximum has to be found
		alpha_neg = lambda x: -alpha(x)
		bounds_an = ((0, 3), (0, 1), (1, 3), (0, 1)) #bounds search space to vicinity of useful values
		#problem: Nelder-Mead (same as fmin) cannot handle bounds -> use L-BFGS-B for now
		new_x=minimize(alpha_neg, [0, 0, 0, 0], method='L-BFGS-B', bounds=bounds_an).x
		#print("new x: " + str(new_x) + " with function value: " + str(alpha(new_x)))
		return new_x

	#needs search space
	def round_x_classification(self, x):
		if len(x) < 4:
			print("Input too short! Returning default values")
			return [0, 0, 1, 0]
		if len(x) > 4:
			print("Input too long! Cropping")
		#rounds the values of new_x to values usable as HPs - the question is what kind of rounding makes sense
		#e.g if lambda should be bounded or more values should be rounded to 0
		lambd = x[0]
		massl = math.trunc(x[1]) #is true if abs val > 1
		min_lv = math.trunc(x[2])
		if(min_lv<1): min_lv = 1
		elif (min_lv>3): min_lv = 3
		one_vs_others = math.trunc(x[3])
		new_x_rd = [lambd, massl, min_lv, one_vs_others]
		return new_x_rd

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
dataset_blobs = deml.datasets.make_blobs(n_samples=samples, n_features=dimension, centers=labels)
data_blobs = deml.DataSet(dataset_blobs, name='Blobs_Set')
#perform_evaluation_at(data_moons, 0.00242204, 0, 1, 0) #.98 evaluation!
#perform_BO_classification(data_blobs, 20)
#perform_RO_classification(data_moons, 10, dimension)
#perform_GO_classification(data_moons, 20)
#perform_evaluation_at(self, data_moons, 0.828603059876013, 0, 2, 1)
def pea_classification(cur_lambd: float, cur_massl: bool, cur_min_lv: int, cur_one_vs_others: bool):
	dataset_blobs = deml.datasets.make_blobs(n_samples=samples, n_features=dimension, centers=labels)
	data_blobs = deml.DataSet(dataset_blobs, name='Blobs_Set')
	cur_data = data_blobs.copy()
	#should implement a smoother way to put in the data set
	classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
	classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
	evaluation = classification.evaluate()
	print("Percentage of correct mappings",evaluation["Percentage correct"])
	#wenn Zeit mit reingerechnet werden soll o.ä. in dieser Funktion
	return evaluation["Percentage correct"]
classification_space = [["interval", 0, 1], ["list", 0, 1], ["list", 1, 2, 3], ["list", 0, 1]]
HPO = HP_Optimization(pea_classification, classification_space)
HPO.perform_evaluation_at(0.00242204, 0, 1, 0)
