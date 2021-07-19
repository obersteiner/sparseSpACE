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

#performs random optimization
def perform_RO_classification(data, amt_it: int, dim: int):
	best_evaluation = 0
	best_x = [1, 1, 1, 1]
	for i in range (0, amt_it, 1):
		x = create_random_x()
		new_eval = perform_evaluation_at(data, x[0], x[1], x[2], x[3])
		if new_eval>best_evaluation:
			best_x = x
			best_evaluation = new_eval
	print("Best evaluation in " + str(amt_it) + " random steps: " + str(best_evaluation) + " at " + str(x))

#returns evaluation for certain Parameters on a certain data set
def perform_evaluation_at(cur_data, cur_lambd: float, cur_massl: bool, cur_min_lv: int, cur_one_vs_others: bool):
	classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
	classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
	evaluation = classification.evaluate()
	print("Percentage of correct mappings",evaluation["Percentage correct"])
	return evaluation["Percentage correct"]

#TODO: will perform Bayesian Optimization; Currently basically performs BO with Naive rounding and a weird beta
#Note: HP Input of classification is (lambd:float, massl:bool, min_lv:int <4, one_vs_others:bool)
def perform_BO_classification(data, amt_it: int, dim: int):
	#notes how many x values currently are in the evidence set - starts with amt_HP+1
	amt_HP = 4
	cur_amt_x: int = amt_HP+1
	x_ret = None
	y_ret = None
	print("I should do Bayesian Optimization for " + str(amt_it) + " iterations!")
	C = create_evidence_set(data, amt_it, amt_HP)
	C_x=C[0]
	C_y=C[1]
	print("Evidence Set: \n" + str(C))
	K_matr = get_cov_matrix(C_x, cur_amt_x)
	print(K_matr)
	#dummy value for beta
	beta = 0.5
	mu = lambda x: get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(C_y))
	sigma_sqrd = lambda x: k(x, x)-get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(get_cov_vector(C_x, x, cur_amt_x)))
	#takes sqrt of abs(sigma_sqrd) bc otherwise fmin gives an error - might be an overall warning sign tho
	sigma = lambda x: math.sqrt(abs(sigma_sqrd(x)))
	alpha = lambda x: mu(x)+(math.sqrt(beta))*sigma(x)
	#negates alpha bc maximum has to be found
	alpha_neg = lambda x: -alpha(x)
	for i in range (0, amt_it, 1):
		print("iteration: " + str(i))
		beta = get_beta(i+1)
		print("beta: " + str(beta))
		#value that will be evaluated and added to the evidence set
		new_x=fmin(alpha_neg, [0, 0, 0, 0]) #Note: Vielleicht kann man die Auswertung bounden, damit keine Werte weit weg vom Möglichen rauskommen
		print("new x: " + str(new_x) + " with function value: " + str(alpha(new_x)))
		#rounds the values of new_x to values usable as HPs - the question is what kind of rounding makes sense, or e.g if lambda should be bounded
		lambd = new_x[0]
		massl = math.trunc(new_x[1]) #is true if abs val > 1
		min_lv = math.trunc(new_x[2])
		if(min_lv<1): min_lv = 1
		elif (min_lv>3): min_lv = 3
		one_vs_others = math.trunc(new_x[3])
		new_x_rd = [lambd, massl, min_lv, one_vs_others]
		print("rounded: " + str(new_x_rd))
		if(not check_if_in_array(new_x_rd, C_x)):
			C_x[cur_amt_x] = new_x_rd
			C_y[cur_amt_x] = perform_evaluation_at(data, new_x_rd[0], new_x_rd[1], new_x_rd[2], new_x_rd[3])
			cur_amt_x += 1
		else: 
			#erstelle neues beta und l - übergebe cur_length / amt_x damit cov stuff darin berechnet werden kann, C_x damit if in ev set geschaut werden kann
			#erstelle neue cov matrix und vector mit neuem l
			#berechne neuen wert mit neuem beta und l
			#ggf ändere vorher schon cov matrix beim testen auf g? weil gebraucht wird für die berechnung
			print("Dead End! I am crying now")
			break
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
	if(is_random):
		for i in range(0, dim_HP+1, 1):
			new_x = create_random_x()
			while(check_if_in_array(new_x, x)):
				new_x = create_random_x()
			x[i] = new_x
			#evaluating takes quite some time, especially for min_lv>1
			y[i] = perform_evaluation_at(data, new_x[0], new_x[1], new_x[2], new_x[3])
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

#TODO: will calculate and return current beta. t is current iteration
def get_beta(t: int):
	r: float = 1.5
	d: int = 4
	delta: float = 0.1
	a: float = 1
	b: float = 1
	beta = 2*math.log(t**2*2*math.pi**2/3*delta)+2*d*math.log(t**2*d*b*r*math.sqrt(math.log(4*d*a/delta)))
	print(beta)
	return beta

#TODO: get l_k and use it
def get_l_k(t: int):
	return 0.5

#TODO: get new beta and l. Currently mock function
def get_new_beta_and_l(cur_beta: float, cur_length, C_x):
	#wsl fnct für get_x mit beta und l sinnvoll - auch für vorher
	#l nachher wieder zurücksetzen - bzw lt?? Was ist das, wie berechnet sich das?
	global l_k #if l_k should be changed permanently
	beta_h = 100
	l_h = 20
	#0 if in ev set C_x, constant otherwise (3)
	p = lambda x: check_if_in_array(x, C_x)*3
	#x[0] is \beta+d\beta, x[1] is l
	g = lambda x: x
	new_beta = (cur_beta+beta_h)/2
	new_l = 10
	return new_beta, new_l

#TODO acquire new x for l, beta, C_x, cur_amt_x, using GP-UCB
def acq_x(l: float, beta: float, C_x, C_y, cur_amt_x):
	print("lol")
	#wait no I need to change the l and the cov function D:
	global l_k
	old_l=l_k
	l_k=l
	K_matr = get_cov_matrix(C_x, cur_amt_x)
	print(K_matr)
	mu = lambda x: get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(C_y))
	sigma_sqrd = lambda x: k(x, x)-get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(get_cov_vector(C_x, x, cur_amt_x)))
	#takes sqrt of abs(sigma_sqrd) bc otherwise fmin gives an error - might be an overall warning sign tho
	sigma = lambda x: math.sqrt(abs(sigma_sqrd(x)))
	alpha = lambda x: mu(x)+(math.sqrt(beta))*sigma(x)
	#negates alpha bc maximum has to be found
	alpha_neg = lambda x: -alpha(x)
	new_x=fmin(alpha_neg, [0, 0, 0, 0]) #Note: Vielleicht kann man die Auswertung bounden, damit keine Werte weit weg vom Möglichen rauskommen
	print("new x: " + str(new_x) + " with function value: " + str(alpha(new_x)))
	return new_x

def round_x_classification(x):
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
#perform_BO_classification(data_moons, 6, dimension)
#print(get_beta(1))
perform_RO_classification(data_moons, 10, dimension)
