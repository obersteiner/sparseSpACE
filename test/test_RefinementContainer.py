import unittest
import sparseSpACE

from sparseSpACE.RefinementObject import *
from sparseSpACE.RefinementContainer import *
from sparseSpACE.combiScheme import *
from sparseSpACE.ErrorCalculator import *


class TestRefinementContainer(unittest.TestCase):

    def test_error_setting(self):
        # initialize container
        ref_objects = []
        grid = TrapezoidalGrid(np.zeros(2), np.ones(2))
        for d in range(100):
            ref_object = RefinementObjectSingleDimension(0,1,0,1,(0,1), grid, 0, 1)
            ref_object.volume = np.array([d])
            ref_objects.append(ref_object)
            ref_object.evaluations = int(d/10 + 1)

        container = RefinementContainer(ref_objects, 1, error_estimator=ErrorCalculatorSingleDimVolumeGuided())

        # calc errors and benefits
        for d in range(100):
            container.calc_error(d, np.inf)
            container.set_benefit(d)

        # check error and benefit values
        for d in range(100):
            self.assertEqual(container.get_error(d), d)
        self.assertEqual(container.get_max_error(), 99)
        self.assertEqual(container.get_max_benefit(), 9.9)
        self.assertEqual(container.get_total_error(), 4950)

    def test_refine(self):
        # initialize container
        ref_objects = []
        grid = TrapezoidalGrid(np.zeros(2), np.ones(2))
        for d in range(100):
            ref_object = RefinementObjectSingleDimension(0, 1, 0, 1, (0, 1), grid, 0, 1, coarsening_level=d)
            ref_object.volume = np.array([d])
            ref_objects.append(ref_object)
        container = RefinementContainer(ref_objects, 1, error_estimator=ErrorCalculatorSingleDimVolumeGuided())
        # calculate errors and benefits
        for d in range(100):
            container.calc_error(d, np.inf)
            container.set_benefit(d)
        # check sizes
        self.assertEqual(container.size(), 100)
        # check get next object for refinement
        is_found, position, next_obj = container.get_next_object_for_refinement(tolerance=0.0)
        self.assertEqual(container.get_object(0), next_obj)
        self.assertTrue(is_found)
        self.assertEqual(position, 0)
        is_found, position, next_obj = container.get_next_object_for_refinement(tolerance=container.get_max_benefit())
        self.assertEqual(container.get_object(99), next_obj)
        self.assertTrue(is_found)
        self.assertEqual(position, 99)
        # no object after last object
        is_found, position, next_obj = container.get_next_object_for_refinement(tolerance=0.0)
        self.assertTrue(not is_found)

        # refine objects
        for d in range(0, 100, 10):
            container.refine(d)

        #check sizes
        self.assertEqual(container.size(), 120)
        self.assertEqual(len(container.get_new_objects()), 20)
        # check content of refinement objects
        for i, ref_obj in enumerate(container.get_new_objects()):
            self.assertTrue(ref_obj.start == 0 or ref_obj.start == 0.5)
            self.assertTrue(ref_obj.end == 0.5 or ref_obj.end == 1)
            self.assertEqual(ref_obj.coarsening_level, max(0,(i//2)*10 - 1))
        # check if apply remove works correct
        container.apply_remove()
        # check sizes
        self.assertEqual(len(container.get_new_objects()), 20)
        for i, ref_obj in enumerate(container.get_new_objects()):
            self.assertTrue(ref_obj.start == 0 or ref_obj.start == 0.5)
            self.assertTrue(ref_obj.end == 0.5 or ref_obj.end == 1)
            self.assertEqual(ref_obj.coarsening_level, max(0,(i//2)*10 - 1))
        # check sizes
        self.assertEqual(container.size(), 110)
        container.clear_new_objects()
        self.assertEqual((len(container.get_new_objects())), 0)
        container.reinit_new_objects()
        self.assertEqual(len(container.get_new_objects()), 110)
        # check additional refinements
        container.clear_new_objects()
        for i, ref_obj in enumerate(container.get_objects()):
            if i < 90:
                self.assertEqual(ref_obj.coarsening_level, (10 * (i //9) + i%9 + 1))
        # refine
        for d in range(0, 110, 9):
            container.refine(d)
        # check sizes
        self.assertEqual(container.size(), 136)
        container.apply_remove()
        self.assertEqual(container.size(), 123)
        for i, ref_obj in enumerate(container.get_objects()):
            if i < 80:
                self.assertEqual(ref_obj.coarsening_level, (10 * (i //8) + i%8 + 2))


    def test_update(self):
        # initialize container
        ref_objects = []
        grid = GlobalTrapezoidalGrid(np.zeros(2), np.ones(2))

        for d in range(100):
            ref_object = RefinementObjectSingleDimension(0,1,0,1,(0,1), grid, 0, 1, coarsening_level=d)
            ref_object.volume = np.array([d])
            ref_objects.append(ref_object)
            ref_object.evaluations = int(d/10 + 1)

        container = RefinementContainer(ref_objects, 1, error_estimator=ErrorCalculatorSingleDimVolumeGuided())
        # check if values are updated correctly
        container.update_values(4)
        for d in range(100):
            self.assertEqual(container.get_object(d).coarsening_level, d+4)

    def test_error_setting_meta_ref(self):
        # initialize container
        containers = []
        for d in range(5):
            ref_objects = []
            grid = GlobalTrapezoidalGrid(np.zeros(d), np.ones(d))
            for n in range(100):
                ref_object = RefinementObjectSingleDimension(0, 1, 0, 1, (0, 1), grid, 0, 1)
                ref_object.volume = np.array([n+d])
                ref_objects.append(ref_object)
                ref_object.evaluations = int((n+d) / 10 + 1)
            container = RefinementContainer(ref_objects, 1, error_estimator=ErrorCalculatorSingleDimVolumeGuided())
            containers.append(container)
        meta_container = MetaRefinementContainer(containers)

        # calc errors and benefits
        meta_container.calc_error(None, np.inf)
        meta_container.set_benefit(None)

        # check errors and benefits
        for d in range(5):
            for n in range(100):
                self.assertEqual(meta_container.get_object((d,n)).error, n+d)
        self.assertEqual(meta_container.get_max_benefit(), 9.9)
        self.assertEqual(meta_container.get_total_error(), 4950*5 + (1+2+3+4) * 100)

    def test_refine_meta(self):
        # initialize container
        containers = []
        for d in range(5):
            ref_objects = []
            grid = GlobalTrapezoidalGrid(np.zeros(d), np.ones(d))
            for n in range(100):
                ref_object = RefinementObjectSingleDimension(0, 1, 0, 1, (0, 1),  grid, 0, 1)
                ref_object.volume = np.array([n + d])
                ref_objects.append(ref_object)
                ref_object.evaluations = int((n + d) / 10 + 1)
            container = RefinementContainer(ref_objects, 1, error_estimator=ErrorCalculatorSingleDimVolumeGuided())
            containers.append(container)
        meta_container = MetaRefinementContainer(containers)

        # calculate error and benefits
        meta_container.calc_error(None, np.inf)
        meta_container.set_benefit(None)
        # check sizes
        for d in range(5):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 100)
        # test get next object for refinement
        is_found, position, next_obj = meta_container.get_next_object_for_refinement(tolerance=0.0)
        self.assertEqual(meta_container.get_object((0,0)), next_obj)
        self.assertTrue(is_found)
        self.assertEqual(position, (0,0))
        is_found, position, next_obj = meta_container.get_next_object_for_refinement(tolerance=container.get_max_benefit())
        self.assertEqual(meta_container.get_object((0,99)), next_obj)
        self.assertTrue(is_found)
        self.assertEqual(position, (0,99))
        is_found, position, next_obj = meta_container.get_next_object_for_refinement(
            tolerance=container.get_max_benefit()+1)
        self.assertTrue(not is_found)
        # no more object after last
        is_found, position, next_obj = meta_container.get_next_object_for_refinement(
            tolerance=0)
        self.assertTrue(not is_found)

        # refine objects
        for d in range(0, 5, 2):
            for n in range(0, 100, 10):
                meta_container.refine((d,n))

        # check sizes in modified containers
        for d in range(0, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 120)
            self.assertEqual(len(container.get_new_objects()), 20)
            # check contents of objects
            for i, ref_obj in enumerate(container.get_new_objects()):
                self.assertTrue(ref_obj.start == 0 or ref_obj.start == 0.5)
                self.assertTrue(ref_obj.end == 0.5 or ref_obj.end == 1)

        # check sizes in non-modifed containers
        for d in range(1, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 100)
            self.assertEqual(len(container.get_new_objects()), 100)

        # check apply remove
        meta_container.apply_remove()

        # check sizes in modified containers
        for d in range(0, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 110)
            self.assertEqual(len(container.get_new_objects()), 20)
            # check contents of objects
            for i, ref_obj in enumerate(container.get_new_objects()):
                self.assertTrue(ref_obj.start == 0 or ref_obj.start == 0.5)
                self.assertTrue(ref_obj.end == 0.5 or ref_obj.end == 1)

        # check sizes in non-modifed containers
        for d in range(1, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 100)
            self.assertEqual(len(container.get_new_objects()), 100)

        # apply additional refinement
        meta_container.reinit_new_objects()
        for d in range(0, 5, 2):
            for n in range(0, 110, 10):
                meta_container.refine((d, n))

        # check sizes in modified containers
        for d in range(0, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 132)
            self.assertEqual(len(container.get_new_objects()), 22)

        # check sizes in non-modified containers
        for d in range(1, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 100)
            self.assertEqual(len(container.get_new_objects()), 100)

        # test apply remove
        meta_container.apply_remove()

        # check sizes after remove
        for d in range(0, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 121)
            self.assertEqual(len(container.get_new_objects()), 22)

        # check sizes after remove
        for d in range(1, 5, 2):
            container = meta_container.get_refinement_container_for_dim(d)
            self.assertEqual(container.size(), 100)
            self.assertEqual(len(container.get_new_objects()), 100)

if __name__ == '__main__':
    unittest.main()