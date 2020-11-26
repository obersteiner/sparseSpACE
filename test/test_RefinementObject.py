import unittest
import sparseSpACE

from sparseSpACE.RefinementObject import *
from sparseSpACE.combiScheme import *

class TestRefinementObject(unittest.TestCase):


    def test_extend_split_object_is_calculated(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d)
            combi_scheme = CombiScheme(dim=d)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid)
            for l in range(1, 10 - d):
                combi_scheme.init_adaptive_combi_scheme(lmin=1, lmax=l)
                combi_grids = combi_scheme.getCombiScheme(lmin=1, lmax=l, do_print=False)

                for component_grid in combi_grids:
                    if l == 1:
                        self.assertTrue(not refinment_object.is_already_calculated(tuple(component_grid.levelvector), tuple(component_grid.levelvector)))
                    else:
                        self.assertEqual(combi_scheme.has_forward_neighbour(component_grid.levelvector), refinment_object.is_already_calculated(tuple(component_grid.levelvector), tuple(component_grid.levelvector + np.ones(d, dtype=int))))
                    refinment_object.add_level(tuple(component_grid.levelvector), tuple(component_grid.levelvector))

    def test_extend_split_object_coarsening_update(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, splitSingleDim=False)
            refinment_object.update(5)
            self.assertEqual(refinment_object.coarseningValue, 5)
            refinment_object.update(5)
            self.assertEqual(refinment_object.coarseningValue, 10)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=3, splitSingleDim=False)
            refinment_object.update(3)
            self.assertEqual(refinment_object.coarseningValue, 6)
            refinment_objects, _, _ = refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 6)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=6, number_of_refinements_before_extend=1, splitSingleDim=False)
            refinment_objects, _, _ = refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 6)
                refinment_objects2, _, _ = ref_obj.refine()
                for ref_obj2 in refinment_objects2:
                    self.assertEqual(ref_obj2.coarseningValue, 5)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=6, number_of_refinements_before_extend=0, splitSingleDim=False)
            refinment_objects, _, _ = refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 5)
                refinment_objects2, _, _ = ref_obj.refine()
                for ref_obj2 in refinment_objects2:
                    self.assertEqual(ref_obj2.coarseningValue, 4)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=0, splitSingleDim=False)
            refinment_objects, _, _= refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 0)

    def test_extend_split_object_contains_points(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid)
            points = get_cross_product_list([np.linspace(a,b,10) for _ in range(d)])
            for p in points:
                self.assertTrue(refinment_object.contains(p))
            points2 = get_cross_product_list([list(np.linspace(b+1,b+3, 10)) + list(np.linspace(a-3, a-1, 10)) for _ in range(d)])
            for p in points2:
                self.assertTrue(not refinment_object.contains(p))
            self.assertEqual(points, refinment_object.subset_of_contained_points(points+points2))

    def test_extend_split_refine(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=6, number_of_refinements_before_extend=1, splitSingleDim=False)
            refinment_objects, increase, update = refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 6)
                for dim in range(d):
                    self.assertTrue(ref_obj.start[dim] > refinment_object.start[dim] or ref_obj.end[dim] < refinment_object.end[dim])
                refinment_object_copy = list(refinment_objects)
                refinment_object_copy.remove(ref_obj)
                for other_obj in refinment_object_copy:
                    middle = 0.5*(ref_obj.end + ref_obj.start)
                    self.assertTrue(not other_obj.contains(middle))
            self.assertEqual(len(refinment_objects), 2**d)
            self.assertEqual(increase, None)
            self.assertEqual(update, None)
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=6, number_of_refinements_before_extend=0, splitSingleDim=False)
            refinment_objects, increase, update = refinment_object.refine()
            for ref_obj in refinment_objects:
                self.assertEqual(ref_obj.coarseningValue, 5)
                for dim in range(d):
                    self.assertTrue(ref_obj.start[dim] == refinment_object.start[dim] or ref_obj.end[dim] == refinment_object.end[dim])
            self.assertEqual(len(refinment_objects), 1)
            self.assertEqual(increase, None)
            self.assertEqual(update, None)
            # test return when coarseningValue == 0
            refinment_object = RefinementObjectExtendSplit(a * np.ones(d), b * np.ones(d), grid, coarseningValue=0, number_of_refinements_before_extend=0, splitSingleDim=False)
            refinment_objects, increase, update = refinment_object.refine()
            self.assertEqual(increase, [1 for _ in range(d)])
            self.assertEqual(update, 1)

    def test_single_dim_refine(self):
        a = -3
        b = 6
        dim = 2
        grid = GlobalTrapezoidalGrid(np.ones(dim)*a, np.ones(dim)*b, dim)
        refinment_object = RefinementObjectSingleDimension(a, b, 0, dim, (0,1), grid, a, b, coarsening_level=2)
        refinment_objects, increase, update = refinment_object.refine()
        self.assertEqual(len(refinment_objects), 2)
        for ref_obj in refinment_objects:
            self.assertEqual(ref_obj.coarsening_level, 1)
        self.assertEqual(increase, None)
        self.assertEqual(update, None)
        self.assertEqual(refinment_objects[0].start, refinment_object.start)
        self.assertEqual(refinment_objects[1].start, 0.5*(refinment_object.end + refinment_object.start))
        self.assertEqual(refinment_objects[0].end, 0.5*(refinment_object.end + refinment_object.start))
        self.assertEqual(refinment_objects[1].end, refinment_object.end)

        refinment_object = RefinementObjectSingleDimension(a, b, 0, 2, (0,1), grid, a, b)
        refinment_objects, increase, update = refinment_object.refine()
        self.assertEqual(len(refinment_objects), 2)
        for ref_obj in refinment_objects:
            self.assertEqual(ref_obj.coarsening_level, 0)
        self.assertEqual(increase, None)
        self.assertEqual(update, None)


if __name__ == '__main__':
    unittest.main()
