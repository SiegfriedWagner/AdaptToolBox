from adaptation.UML import LogitUML, UMLParameter, UML, WeibullUML, GaussianUML
import unittest
from numpy.testing import assert_allclose
from support import loadmat, debug_on

class UMLParameterTest(unittest.TestCase):

    def test_constructor_error(self):
        with self.assertRaises(ValueError):
            tested = UMLParameter(value=10, scale='lin')
            
    def test_constructor(self):
        tested = UMLParameter(scale="lin",
                             dist="flat",
                             min_value=-10,
                             max_value=30,
                             mu=0,
                             std=20,
                             n=61)
        self.assertEqual(tested.scale, "lin")
        self.assertEqual(tested.dist, "flat")
        self.assertEqual(tested.min_value, -10)
        self.assertEqual(tested.max_value, 30)
        self.assertEqual(tested.mu, 0)
        self.assertEqual(tested.std, 20)
        self.assertEqual(tested.n, 61)

    def test_setParSpace_lin(self):
        target = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tested = UMLParameter(scale="lin",
                             dist="flat",
                             min_value=-10,
                             max_value=10,
                             mu=0,
                             std=20,
                             n=21)
        tested.setParSpace()
        self.assertSequenceEqual(list(tested.space), target)

    def test_setParSpace_log(self):
       target =[1, 1.12443722756960, 1.26435907874441, 1.42169241715583, 1.59860388000342, 1.79752971481306, 2.02120932899837, 2.27272301423675, 2.55553436516200, 2.87353797652160, 3.23111307563592, 3.63318382873194, 4.08528715163006, 4.59364895860462, 5.16526989944137, 5.80802176537658, 6.53075589152395, 7.34342504859904, 8.25722050251188, 9.28472612927533, 10.4400917075454, 11.7392277752048, 13.2000247333593, 14.8425992150287, 16.6895711112737, 18.7663750696863, 21.1016107548893, 23.7274366944807, 26.6800131340751, 30.0000000000000]
       tested = UMLParameter(scale="log",
                             dist="flat",
                             min_value=1,
                             max_value=30,
                             mu=0,
                             std=20,
                             n=30)
       tested.setParSpace()
       assert_allclose(tested.space, target, rtol=0, atol=1e-10)

    def test_setPrior_flat(self):
        pass

    def test_setPrior_norm(self):
        pass
        

class UMLTestCase(unittest.TestCase):
    class GenericUML(UML):
        def _calc_sweetpoints(self, phi):
            return [0, 0, 0]

        def _prob_function(self, x, alpha, beta, gamma, lamb):
            return 1

    def setUp(self):
        alpha = UMLParameter(scale="lin",
                             dist="flat",
                             min_value=-10,
                             max_value=30,
                             mu=0,
                             std=20,
                             n=61)
        beta = UMLParameter(scale="log",
                            dist="norm",
                            min_value=0.1,
                            max_value=10,
                            mu=0,
                            std=2,
                            n=41)

        gamma = UMLParameter(value=0.5)

        lamb = UMLParameter(scale="lin",
                            dist="flat",
                            min_value=0,
                            max_value=0.2,
                            mu=0,
                            std=0.1,
                            n=11)
        self.tested = UMLTestCase.GenericUML(safemode=False,
                                             max_stimuli=30,
                                             min_stimuli=-30,
                                             value=30,
                                             method='mean',
                                             alpha=alpha,
                                             beta=beta,
                                             gamma=gamma,
                                             lamb=lamb)
        self.uml_source_data = loadmat('./test/GenericUMLData.mat')
        self.uml_source_data = self.uml_source_data['ans']
        
    def tearDown(self):
        del(self.tested)
        del(self.uml_source_data)
        
    def test_setP0(self):
        self.tested._setP0()
        alpha = self.uml_source_data['a']
        beta = self.uml_source_data['b']
        lamb = self.uml_source_data['l']
        assert_allclose(self.tested.alpha.meshgrid, alpha, rtol=0, atol=1e-7)
        assert_allclose(self.tested.beta.meshgrid, beta, rtol=0, atol=1e-7)
        assert_allclose(self.tested.lamb.meshgrid, lamb, rtol=0, atol=1e-7)

        A = self.tested.alpha.setPrior()
        B = self.tested.beta.setPrior()
        L = self.tested.lamb.setPrior()
        assert_allclose(A, self.uml_source_data['A'], rtol=0, atol=1e-7)
        assert_allclose(B, self.uml_source_data['B'], rtol=0, atol=1e-7)
        assert_allclose(L, self.uml_source_data['L'], rtol=0, atol=1e-7)

        ABL = self.uml_source_data['ABL']
        assert_allclose(A * B * L, ABL, rtol=0, atol=1e-07)
        
        ABLprob = self.uml_source_data['ABLprob']
        assert_allclose(self.tested._prepare_prob(A * B * L), ABLprob, rtol=0, atol=1e-7)
        p = self.uml_source_data['p']
        assert_allclose(self.tested.p, p, rtol=0, atol=1e-7)
        
class LogitUMLTestCase(unittest.TestCase):

    def setUp(self):
        alpha = UMLParameter(scale="lin",
                             dist="flat",
                             min_value=-10,
                             max_value=30,
                             mu=0,
                             std=20,
                             n=61)
        beta = UMLParameter(scale="log",
                            dist="norm",
                            min_value=0.1,
                            max_value=10,
                            mu=0,
                            std=2,
                            n=41)

        gamma = UMLParameter(value=0.5)

        lamb = UMLParameter(scale="lin",
                            dist="flat",
                            min_value=0,
                            max_value=0.2,
                            mu=0,
                            std=0.1,
                            n=11)

        self.tested = LogitUML(safemode=False,
                               max_stimuli=30,
                               min_stimuli=-30,
                               value=30,
                               method='mean',
                               alpha=alpha,
                               beta=beta,
                               gamma=gamma,
                               lamb=lamb)
        self.uml_source_data = loadmat('./test/LogitUMLData.mat')
        self.uml_source_data = self.uml_source_data['ans']

    def tearDown(self):
        del(self.tested)

    def test_update_single(self):
        self.tested.update(1)
        assert_allclose(self.tested.phi,
                        [[9.7232, 2.2039, 0.5000, 0.0956]], rtol=0, atol=1e-03)
        
        
    def test_final_run(self):
        responses = self.uml_source_data['r']
        for response in responses:
            self.tested.update(response, incorrect_set=(0,))
        assert_allclose(self.tested.phi, self.uml_source_data['phi'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.swpts, self.uml_source_data['swpts'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)

class WeibullUMLTestCase(unittest.TestCase):

    def setUp(self):
        alpha = UMLParameter(min_value=1e-5,
                             max_value=1e5,
                             n=61,
                             scale='log',
                             dist='flat',
                             mu=-2,
                             std=2)
        beta = UMLParameter(min_value=1,
                            max_value=20,
                            n=11,
                            scale='log',
                            dist='norm',
                            mu=0,
                            std=1)
        gamma = UMLParameter(value = 0.5)
        lamb = UMLParameter(scale="lin",
                            dist="flat",
                            min_value=0,
                            max_value=0.2,
                            mu=0,
                            std=0.1,
                            n=11)
        self.tested = WeibullUML(safemode=False,
                                 max_stimuli=1e5,
                                 min_stimuli=1e-5,
                                 value=1e5,
                                 method='mean',
                                 alpha=alpha,
                                 beta=beta,
                                 gamma=gamma,
                                 lamb=lamb)
        self.uml_source_data = loadmat('./test/WeibullUMLData.mat')
        self.uml_source_data = self.uml_source_data['ans']

    def tearDown(self):
        del(self.tested)
        del(self.uml_source_data)

    def test_constructor(self):
        par = self.uml_source_data['par']
        self.assertAlmostEqual(self.tested.min_stimuli, par['x_lim'][0], places=4)
        self.assertEqual(self.tested.max_stimuli, par['x_lim'][1])
        self.assertEqual(self.tested.value, par['x0'])
        self.assertEqual(self.tested.method, par['method'])
        alpha = self.tested.alpha
        beta = self.tested.beta
        gamma = self.tested.gamma
        lamb = self.tested.lamb
        self.assertAlmostEqual(alpha.min_value, par['alpha']['limits'][0], places=4)
        self.assertEqual(alpha.max_value, par['alpha']['limits'][1])
        self.assertEqual(alpha.scale, par['alpha']['scale'])
        self.assertEqual(alpha.dist, par['alpha']['dist'])
        self.assertEqual(alpha.mu, par['alpha']['mu'])
        self.assertEqual(alpha.std, par['alpha']['std'])

        self.assertAlmostEqual(beta.min_value, par['beta']['limits'][0], places=4)
        self.assertEqual(beta.max_value, par['beta']['limits'][1])
        self.assertEqual(beta.scale, par['beta']['scale'])
        self.assertEqual(beta.dist, par['beta']['dist'])
        self.assertEqual(beta.mu, par['beta']['mu'])
        self.assertEqual(beta.std, par['beta']['std'])

        self.assertEqual(gamma.value, par['gamma'])

        self.assertAlmostEqual(lamb.min_value, par['lambda']['limits'][0], places=4)
        self.assertEqual(lamb.max_value, par['lambda']['limits'][1])
        self.assertEqual(lamb.scale, par['lambda']['scale'])
        self.assertEqual(lamb.dist, par['lambda']['dist'])
        self.assertEqual(lamb.mu, par['lambda']['mu'])
        self.assertEqual(lamb.std, par['lambda']['std'])
        
    def test_update_single(self):
        self.tested.update(1)
        assert_allclose(self.tested.phi,
                        [[4845.41789172302, 5.32777953244399, 0.500000000000000, 0.0955748081756037]], rtol=0, atol=1e-03)
        
    def test_final_run(self):
        responses = self.uml_source_data['r']
        for response in responses:
            self.tested.update(response, incorrect_set=(0,))
        assert_allclose(self.tested.phi, self.uml_source_data['phi'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=1, atol=0)


class GaussianUMLTestCase(unittest.TestCase):

    def setUp(self):
        alpha = UMLParameter(min_value=-20,
                             max_value=20,
                             n=61,
                             scale='lin',
                             dist='flat',
                             mu=0,
                             std=20)
        beta = UMLParameter(min_value=1,
                            max_value=20,
                            n=41,
                            scale='log',
                            dist='norm',
                            mu=0.5,
                            std=2)
        gamma = UMLParameter(value = 0.5)
        lamb = UMLParameter(scale="lin",
                            dist="flat",
                            min_value=0,
                            max_value=0.1,
                            mu=0,
                            std=0.1,
                            n=5)
        self.tested = GaussianUML(safemode=False,
                                  max_stimuli=30,
                                  min_stimuli=-30,
                                  value=25,
                                  method='mean',
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  lamb=lamb)
        self.uml_source_data = loadmat('./test/GaussianUMLData.mat')
        self.uml_source_data = self.uml_source_data['ans']

    def tearDown(self):
        del(self.tested)
        del(self.uml_source_data)

    def test_constructor(self):
        par = self.uml_source_data['par']
        self.assertAlmostEqual(self.tested.min_stimuli, par['x_lim'][0], places=4)
        self.assertEqual(self.tested.max_stimuli, par['x_lim'][1])
        self.assertEqual(self.tested.value, par['x0'])
        self.assertEqual(self.tested.method, par['method'])
        alpha = self.tested.alpha
        beta = self.tested.beta
        gamma = self.tested.gamma
        lamb = self.tested.lamb
        self.assertAlmostEqual(alpha.min_value, par['alpha']['limits'][0], places=4)
        self.assertEqual(alpha.max_value, par['alpha']['limits'][1])
        self.assertEqual(alpha.scale, par['alpha']['scale'])
        self.assertEqual(alpha.dist, par['alpha']['dist'])
        self.assertEqual(alpha.mu, par['alpha']['mu'])
        self.assertEqual(alpha.std, par['alpha']['std'])

        self.assertAlmostEqual(beta.min_value, par['beta']['limits'][0], places=4)
        self.assertEqual(beta.max_value, par['beta']['limits'][1])
        self.assertEqual(beta.scale, par['beta']['scale'])
        self.assertEqual(beta.dist, par['beta']['dist'])
        self.assertEqual(beta.mu, par['beta']['mu'])
        self.assertEqual(beta.std, par['beta']['std'])

        self.assertEqual(gamma.value, par['gamma'])

        self.assertAlmostEqual(lamb.min_value, par['lambda']['limits'][0], places=4)
        self.assertEqual(lamb.max_value, par['lambda']['limits'][1])
        self.assertEqual(lamb.scale, par['lambda']['scale'])
        self.assertEqual(lamb.dist, par['lambda']['dist'])
        self.assertEqual(lamb.mu, par['lambda']['mu'])
        self.assertEqual(lamb.std, par['lambda']['std'])
        
    def test_final_run(self):
        responses = self.uml_source_data['r']
        for response in responses:
            self.tested.update(response, incorrect_set=(0,))

        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
    
