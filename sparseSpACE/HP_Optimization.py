import sparseSpACE.DEMachineLearning as deml

print("I am working!")

#Basically copied code from Tutorial_DEMachineLearning
def perform_optimization_mock():
	#prepare Dataset
	samples = 500
	dimension = 2
	labels = 6
	sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15)
	data = deml.DataSet(sklearn_dataset, name='Input_Set')
	#for storing the current best evaluation
	best_evaluation = None
	#best_evaluation = 0.7
	print("original best evaluation: " + str(best_evaluation))
	#todo: split Dataset
	classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
	classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True) #True
	evaluation = classification.evaluate()
	print("Percentage of correct mappings",evaluation["Percentage correct"])
	#currently only looking at Percentage correct for evaluating
	current_evaluation = evaluation["Percentage correct"]
	if best_evaluation == None or current_evaluation < best_evaluation:
		best_evaluation = current_evaluation
		print("Best evaluation is now " + str(best_evaluation))
	else:	print("Best evaluation is still " + str(best_evaluation))
	
	print("In the end, best evaluation is " + str(best_evaluation))


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

perform_optimization_mock()
print("I was working!")
