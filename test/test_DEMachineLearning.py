import unittest
import sparseSpACE
import sparseSpACE.DEMachineLearning as deml
import numpy as np

class TestDEMachineLearning(unittest.TestCase):

    def test_dataset(self):
        raw_data_samples = np.array([[-6.59997755e-01,  7.65980127e-01],
                                     [ 1.98673591e+00,  3.36458863e-01],
                                     [ 1.50372162e+00, -4.01039567e-01],
                                     [ 1.65881384e+00, -2.98364448e-01],
                                     [ 5.47317705e-01,  7.98227910e-01],
                                     [ 7.87056115e-01, -4.67993057e-01],
                                     [-6.06835318e-01,  9.21651213e-01],
                                     [ 1.06572897e+00,  9.69165356e-02],
                                     [ 4.88777881e-01, -4.01428781e-01],
                                     [ 4.04771616e-01, -1.46416166e-01],
                                     [ 1.69921522e+00, -1.52812127e-01],
                                     [ 7.42023866e-01,  7.01429162e-01],
                                     [-9.99678059e-01, -1.36051736e-02],
                                     [ 1.97076650e+00,  2.31107608e-01],
                                     [ 8.03159347e-03,  3.17271053e-01],
                                     [ 8.84929326e-02, -5.57975593e-02],
                                     [ 1.91778358e+00,  4.02405202e-01],
                                     [-8.98335509e-01,  4.81082660e-01],
                                     [ 2.33050481e-01,  1.05617044e+00],
                                     [ 8.80122429e-01,  4.91610576e-01],
                                     [ 9.12813676e-01,  2.84665016e-01],
                                     [-5.06325666e-02,  4.67381188e-01],
                                     [-2.69247098e-01,  9.45235308e-01],
                                     [ 9.95389846e-01, -4.94658577e-01],
                                     [ 1.12949667e+00, -4.86798871e-01],
                                     [-1.92182344e-01,  9.80413533e-01],
                                     [-1.07218961e+00,  1.39604402e-01],
                                     [ 9.53308332e-02,  1.01596105e+00],
                                     [ 7.47454325e-02,  1.97825566e-03],
                                     [ 1.80510715e+00, -1.19978052e-01],
                                     [ 8.02345135e-01,  5.97528328e-01],
                                     [ 2.63927046e-01, -1.01881038e-01],
                                     [ 2.41040104e-02,  2.14648340e-01],
                                     [-4.27355120e-01,  1.01848856e+00],
                                     [-9.86486899e-01,  2.95959258e-01],
                                     [ 9.12153394e-01,  4.39446288e-01],
                                     [ 2.43454926e-01,  8.80678773e-01],
                                     [-8.91966325e-01,  3.47641156e-01],
                                     [ 3.94638751e-01, -3.40023801e-01],
                                     [ 1.21839455e+00, -5.06204755e-01],
                                     [-2.80862186e-02,  9.85037095e-01],
                                     [ 1.90405088e+00,  2.35064389e-02],
                                     [ 4.97716995e-01,  8.90922187e-01],
                                     [ 1.41444792e+00, -4.84225249e-01],
                                     [-7.78848408e-01,  7.00152837e-01],
                                     [ 8.17336783e-01, -5.81295929e-01],
                                     [ 1.90034923e+00,  9.15028796e-02],
                                     [ 5.23524094e-01, -4.45336380e-01],
                                     [-5.57049370e-01,  8.29379966e-01],
                                     [ 1.05386957e+00,  2.74580757e-02]], dtype=np.float64)
        raw_data_labels = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                                    0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                                    0, 1, 1, 1, 0, 0], dtype=np.int64)
        raw_data = (raw_data_samples, raw_data_labels)

        # test constructor
        data = deml.DataSet(raw_data)
        self.assertEqual(len(raw_data[0]), len(data.get_data()[0]))
        for raw, dat in zip(raw_data[0], data.get_data()[0]):
            self.assertEqual(len(raw), len(dat))
            for x, y in zip(raw, dat):
                self.assertEqual(x, y)
        for raw_s, dat_s in zip(raw_data[1], data.get_data()[1]):
            self.assertEqual(raw_s, dat_s)

        # test scaling
        data.scale_range((0, 1))
        for d_min, d_max in zip(data.get_min_data(), data.get_max_data()):
            self.assertAlmostEqual(d_min, 0)
            self.assertAlmostEqual(d_max, 1)
        data.scale_factor(-2)
        for d_min, d_max in zip(data.get_min_data(), data.get_max_data()):
            self.assertAlmostEqual(d_min, -2)
            self.assertAlmostEqual(d_max, 0)
        data.shift_value(5)
        for d_min, d_max in zip(data.get_min_data(), data.get_max_data()):
            self.assertAlmostEqual(d_min, 3)
            self.assertAlmostEqual(d_max, 5)
        data.revert_scaling()
        for raw, dat in zip(raw_data[0], data.get_data()[0]):
            for x, y in zip(raw, dat):
                self.assertAlmostEqual(x, y)

        # test splitting
        part0, part1 = data.split_labels()
        for part0_s, part1_s in zip(part0[1], part1[1]):
            self.assertEqual(part0_s, 0)
            self.assertEqual(part1_s, 1)
        data = part0.concatenate(part1)
        self.assertEqual(data.get_length(), 50)
        data.remove_labels(0.3)
        without_labels, with_labels = data.split_without_labels()
        self.assertEqual(without_labels.get_length(), 15)
        part0, part1 = without_labels.split_pieces(0.4)
        self.assertEqual(part0.get_length(), 6)

        # miscellaneous
        data.remove_samples([0, 1, 2, 3, 4])
        self.assertEqual(data.get_length(), 45)

    def test_classification(self):
        raw_data_samples = np.array([[-6.59997755e-01, 7.65980127e-01],
                                     [1.98673591e+00, 3.36458863e-01],
                                     [1.50372162e+00, -4.01039567e-01],
                                     [1.65881384e+00, -2.98364448e-01],
                                     [5.47317705e-01, 7.98227910e-01],
                                     [7.87056115e-01, -4.67993057e-01],
                                     [-6.06835318e-01, 9.21651213e-01],
                                     [1.06572897e+00, 9.69165356e-02],
                                     [4.88777881e-01, -4.01428781e-01],
                                     [4.04771616e-01, -1.46416166e-01],
                                     [1.69921522e+00, -1.52812127e-01],
                                     [7.42023866e-01, 7.01429162e-01],
                                     [-9.99678059e-01, -1.36051736e-02],
                                     [1.97076650e+00, 2.31107608e-01],
                                     [8.03159347e-03, 3.17271053e-01],
                                     [8.84929326e-02, -5.57975593e-02],
                                     [1.91778358e+00, 4.02405202e-01],
                                     [-8.98335509e-01, 4.81082660e-01],
                                     [2.33050481e-01, 1.05617044e+00],
                                     [8.80122429e-01, 4.91610576e-01],
                                     [9.12813676e-01, 2.84665016e-01],
                                     [-5.06325666e-02, 4.67381188e-01],
                                     [-2.69247098e-01, 9.45235308e-01],
                                     [9.95389846e-01, -4.94658577e-01],
                                     [1.12949667e+00, -4.86798871e-01],
                                     [-1.92182344e-01, 9.80413533e-01],
                                     [-1.07218961e+00, 1.39604402e-01],
                                     [9.53308332e-02, 1.01596105e+00],
                                     [7.47454325e-02, 1.97825566e-03],
                                     [1.80510715e+00, -1.19978052e-01],
                                     [8.02345135e-01, 5.97528328e-01],
                                     [2.63927046e-01, -1.01881038e-01],
                                     [2.41040104e-02, 2.14648340e-01],
                                     [-4.27355120e-01, 1.01848856e+00],
                                     [-9.86486899e-01, 2.95959258e-01],
                                     [9.12153394e-01, 4.39446288e-01],
                                     [2.43454926e-01, 8.80678773e-01],
                                     [-8.91966325e-01, 3.47641156e-01],
                                     [3.94638751e-01, -3.40023801e-01],
                                     [1.21839455e+00, -5.06204755e-01],
                                     [-2.80862186e-02, 9.85037095e-01],
                                     [1.90405088e+00, 2.35064389e-02],
                                     [4.97716995e-01, 8.90922187e-01],
                                     [1.41444792e+00, -4.84225249e-01],
                                     [-7.78848408e-01, 7.00152837e-01],
                                     [8.17336783e-01, -5.81295929e-01],
                                     [1.90034923e+00, 9.15028796e-02],
                                     [5.23524094e-01, -4.45336380e-01],
                                     [-5.57049370e-01, 8.29379966e-01],
                                     [1.05386957e+00, 2.74580757e-02]], dtype=np.float64)
        raw_data_labels = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                                    0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                                    0, 1, 1, 1, 0, 0], dtype=np.int64)
        raw_data = (raw_data_samples, raw_data_labels)
        data = deml.DataSet(raw_data, "Snake_Set")

        # test with initial learning/testing ratio
        class_object = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
        class_object.perform_classification(masslumping=True, minimum_level=1, maximum_level=5, lambd=0.0, print_metrics=False)
        evaluation = class_object.evaluate()
        self.assertEqual(evaluation.get("Wrong mappings"), 0)
        self.assertEqual(evaluation.get("Total mappings"), 10)
        self.assertEqual(evaluation.get("Percentage correct"), 1.0)

        # test classification of new data
        test_data_samples = np.array([[ 4.99505733e-01, -3.92247458e-01],
                                      [ 1.64520521e+00, -2.16953653e-01],
                                      [ 1.58942130e-01,  1.07962269e+00],
                                      [ 3.83260542e-01,  7.97982871e-01],
                                      [ 2.03785226e+00,  4.90876995e-01],
                                      [ 9.25938129e-01,  1.43358641e-01],
                                      [-8.49493051e-01,  5.33078453e-01],
                                      [ 1.01657428e+00, -4.64200772e-01],
                                      [ 1.23847285e+00, -4.33601110e-01],
                                      [ 9.49535619e-01,  3.99540324e-01]], dtype=np.float64)
        test_data_labels = np.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
        test_data = (test_data_samples, test_data_labels)
        data_t = deml.DataSet(test_data, "Test_Snake")
        # 2 test samples are out of bounds, so only 8 will be tested
        # 2 samples' classes were swapped, so the number of wrong mappings should be 2
        evaluation_t = class_object.test_data(data_t, print_output=False, print_removed=False, print_incorrect_points=False)
        self.assertEqual(evaluation_t.get("Wrong mappings"), 2)
        self.assertEqual(evaluation_t.get("Total mappings"), 8)
        self.assertEqual(evaluation_t.get("Percentage correct"), 0.75)

    def test_clustering(self):
        raw_data_samples = np.array([[-6.59997755e-01, 7.65980127e-01],
                                     [1.98673591e+00, 3.36458863e-01],
                                     [1.50372162e+00, -4.01039567e-01],
                                     [1.65881384e+00, -2.98364448e-01],
                                     [5.47317705e-01, 7.98227910e-01],
                                     [7.87056115e-01, -4.67993057e-01],
                                     [-6.06835318e-01, 9.21651213e-01],
                                     [1.06572897e+00, 9.69165356e-02],
                                     [4.88777881e-01, -4.01428781e-01],
                                     [4.04771616e-01, -1.46416166e-01],
                                     [1.69921522e+00, -1.52812127e-01],
                                     [7.42023866e-01, 7.01429162e-01],
                                     [-9.99678059e-01, -1.36051736e-02],
                                     [1.97076650e+00, 2.31107608e-01],
                                     [8.03159347e-03, 3.17271053e-01],
                                     [8.84929326e-02, -5.57975593e-02],
                                     [1.91778358e+00, 4.02405202e-01],
                                     [-8.98335509e-01, 4.81082660e-01],
                                     [2.33050481e-01, 1.05617044e+00],
                                     [8.80122429e-01, 4.91610576e-01],
                                     [9.12813676e-01, 2.84665016e-01],
                                     [-5.06325666e-02, 4.67381188e-01],
                                     [-2.69247098e-01, 9.45235308e-01],
                                     [9.95389846e-01, -4.94658577e-01],
                                     [1.12949667e+00, -4.86798871e-01],
                                     [-1.92182344e-01, 9.80413533e-01],
                                     [-1.07218961e+00, 1.39604402e-01],
                                     [9.53308332e-02, 1.01596105e+00],
                                     [7.47454325e-02, 1.97825566e-03],
                                     [1.80510715e+00, -1.19978052e-01],
                                     [8.02345135e-01, 5.97528328e-01],
                                     [2.63927046e-01, -1.01881038e-01],
                                     [2.41040104e-02, 2.14648340e-01],
                                     [-4.27355120e-01, 1.01848856e+00],
                                     [-9.86486899e-01, 2.95959258e-01],
                                     [9.12153394e-01, 4.39446288e-01],
                                     [2.43454926e-01, 8.80678773e-01],
                                     [-8.91966325e-01, 3.47641156e-01],
                                     [3.94638751e-01, -3.40023801e-01],
                                     [1.21839455e+00, -5.06204755e-01],
                                     [-2.80862186e-02, 9.85037095e-01],
                                     [1.90405088e+00, 2.35064389e-02],
                                     [4.97716995e-01, 8.90922187e-01],
                                     [1.41444792e+00, -4.84225249e-01],
                                     [-7.78848408e-01, 7.00152837e-01],
                                     [8.17336783e-01, -5.81295929e-01],
                                     [1.90034923e+00, 9.15028796e-02],
                                     [5.23524094e-01, -4.45336380e-01],
                                     [-5.57049370e-01, 8.29379966e-01],
                                     [1.05386957e+00, 2.74580757e-02]], dtype=np.float64)
        raw_data_labels = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                                    0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                                    0, 1, 1, 1, 0, 0], dtype=np.int64)
        raw_data = (raw_data_samples, raw_data_labels)
        data = deml.DataSet(raw_data, "Cobra_Set")

        # test clustering
        clus_object = deml.Clustering(data, number_nearest_neighbors=15, edge_cutting_threshold=0.4)
        clus_object.perform_clustering(masslumping=True, minimum_level=1, maximum_level=5, lambd=0.0, print_metrics=False)
        evaluation = clus_object.evaluate()
        self.assertEqual(evaluation.get("Wrong mappings"), 22)
        self.assertEqual(evaluation.get("Total mappings"), 50)
        self.assertEqual(evaluation.get("Percentage correct"), 0.56)


if __name__ == '__main__':
    unittest.main()
