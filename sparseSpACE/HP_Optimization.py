import sparseSpACE.DEMachineLearning as deml
import numpy
import scipy
import math

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
def perform_BO_classification(data, amt_it: int):
	print("I should do Bayesian Optimization for " + str(amt_it) + " iteration!")
	#dummy value for beta
	beta = 0.5
	#use squared exp. kernel as covariance function, with parameters sigma_k and l_k
	sigma_k = 1
	l_k = 0.5
	k = lambda x: x
	mu = lambda x: x**2+4
	sigma = lambda x: x**2
	alpha = lambda x: mu(x)+(beta**(0.5))*sigma(x)
	for i in range (0, amt_it, 1):
		beta = get_beta(i)
		print("beta: " + str(beta) + " mu(x): " + str(mu(2)) + " sigma(x): " + str(sigma(2)))
		print(alpha(2))

#TODO: will calculate and return current beta
def get_beta(cur_it: int):
	beta = cur_it/2
	return beta

#print("I am working!")

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
perform_BO_classification(data_moons, 5)
#print("I was working!")
