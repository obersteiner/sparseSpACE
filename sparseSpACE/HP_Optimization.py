import sparseSpACE.DEMachineLearning as deml

print("I am working!")

#Basically copied code from Tutorial_DEMachineLearning
def perform_optimization_grid_simple():
	#prepare Dataset
	samples = 500
	dimension = 2
	labels = 6
	sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15)
	data = deml.DataSet(sklearn_dataset, name='Input_Set')
	#for storing the current best evaluation
	best_evaluation = None
	best_time = None
	#storing the parameters of the best evaluation
	best_lambd = None
	best_masslump = None
	#best_evaluation = 0.7
	print("original best evaluation: " + str(best_evaluation))
	#todo: split Dataset
	classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
	classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True)
	#perform grid search for a few values of masslumping and lambd
	for i in range (0, 11, 10):
		for j in [True, False]:
			#original settings: classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True)
			classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
			current_lambd = i/10
			current_masslump = j
			print("masslumping = " + str(current_masslump) + "; lambd = " + str(current_lambd))
			classification.perform_classification(masslumping=current_masslump, lambd=current_lambd, minimum_level=1, maximum_level=5, print_metrics=False)
			current_time = classification._time_used
			print ("current time needed = " + str(current_time))
			evaluation = classification.evaluate()
			print("Percentage of correct mappings",evaluation["Percentage correct"])
			#currently only looking at Percentage correct and time needed for evaluating
			current_evaluation = evaluation["Percentage correct"]
			if best_evaluation == None or current_evaluation > best_evaluation or (current_evaluation == best_evaluation and (best_time == None or best_time>current_time)):
				best_evaluation = current_evaluation
				best_lambd = current_lambd
				best_time = current_time
				best_masslump = current_masslump
				print("Best evaluation is now " + str(best_evaluation) + " with masslump = " + str (best_masslump) + ", lambd = " + str(best_lambd) + " and time = " + str(best_time))
			else:	print("Best evaluation is still " + str(best_evaluation) + " with masslump = " + str (best_masslump) + " and lambd = " + str(best_lambd)+ " and time = " + str(best_time))
			print ("END OF CURRENT EVALUATION \n")
	print("In the end, best evaluation is " + str(best_evaluation)  + " with masslump = " + str (best_masslump) + " and lambd = " + str(best_lambd)+ " and time = " + str(best_time))


	"""# also more testing data can be added and the results can be printed immediately
	data_copy = data.copy()                                              # deepcopied
	data_copy.scale_range((0.005, 0.995))                                # scaled
	part0, part1 = data_copy.split_pieces(0.5)                           # split
	data_copy = part0.concatenate(part1)                                 # concatenated
	data_copy.set_name('2nd_Set')                                        # renamed
	data_copy.remove_labels(0.5)                                         # freed of some label assignments to samples
	without_labels, with_labels = data_copy.split_without_labels()
	with_labels.set_name("Test_new_data")
	classification.test_data(with_labels, print_output=True)"""

perform_optimization_grid_simple()
print("I was working!")
