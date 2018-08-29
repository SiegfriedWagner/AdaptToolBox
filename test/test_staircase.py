import unittest
from AdaptToolBox.adaptation.Staircase import _Staircase, LinearStaircase


class StaircaseTestCase(unittest.TestCase):
    
    class GenericStaircase(_Staircase):

        def up(self):
            pass

        def down(self):
            pass

        def no_change(self):
            pass
        
    def setUp(self):
        ndown = 3
        nup = 2
        n = 0
        self.tested = StaircaseTestCase.GenericStaircase(reset_after_change=False,
                                                         ndown=ndown, nup=nup,
                                                         n=n)
        
    def tearDown(self):
        assert(self.tested)
        del(self.tested)
    
    def test_init(self):
        self.assertEqual(self.tested.ndown, 3)
        self.assertEqual(self.tested.nup, 2)
        self.assertEqual(self.tested.n, 0)

    def test_sequence(self):
        responses = (1, 1, 1, -1, 1, -1, 1, 1, -1, -1)
        targets =  (1, 2, 2, -1, 1, -1, 1, 2, -1, -1)
        outputs = []
        for response in responses:
            self.tested.update(response)
            outputs.append(self.tested.n)
        self.assertSequenceEqual(targets, outputs)


class LinearStaircaseTestCase(unittest.TestCase):

    def setUp(self):
        self.tested = LinearStaircase(reset_after_change=False,
                                     ndown=3,
                                     nup=2,
                                     n=0,
                                     diff_up=1.0,
                                     diff_down=1.0,
                                     value=5.0,
                                     max_value=10.0,
                                      min_value=0.0)

    def tearDown(self):
        del(self.tested)

    def test_sequence(self):
        responses = (1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1)
        targets =   (1, 2, 2, -1, 1, -1, 1, 2, -1, -1, -1, -1, -1)
        outputs = []
        for response in responses:
            self.tested.update(response)
            outputs.append(self.tested.n)
        self.assertSequenceEqual(targets, outputs)

    def test_values(self):
        responses = (1, 1, 1, 1, 1, -1, 1, 1, -1, -1)
        targets =   (5, 5, 4, 3, 2,  2, 2, 2,  2,  3)
        outputs = []
        for response in responses:
            self.tested.update(response)
            outputs.append(self.tested.value)
        self.assertSequenceEqual(targets, outputs)

    def test_sequence_2(self):
        self._tested = LinearStaircase(reset_after_change=True,
                                      ndown=2,
                                      nup=1,
                                      n=0,
                                      diff_up=1.0,
                                      diff_down=1.0,
                                      value=3.0,
                                      max_value=3.0,
                                       min_value=0.0)
        
        responses = (1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
        matlab_target = [4, 4, 3, 3, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 2, 3, 4, 4, 3, 3, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 2, 1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 2, 2, 1, 2, 2, 1, 2, 3, 3, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 4, 4, 3, 3, 2, 2, 1, 2, 3, 3, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1]
        outputs = []
        for response in responses:
            outputs.append(self._tested.value+1)
            self._tested.update(response, incorrect_set=(0,))
        self.assertSequenceEqual(matlab_target, outputs)
if __name__ == '__main__':
    unittest.main()
