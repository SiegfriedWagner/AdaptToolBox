from adaptation.UML import LogitUML, UMLParameter, _UML, WeibullUML, GaussianUML
import unittest, code
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
    class GenericUML(_UML):
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
        assert_allclose(self.tested.x, self.uml_source_data['x'], rtol=0, atol=1e-7)
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
        assert_allclose(self.tested.x, self.uml_source_data['x'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.phi, self.uml_source_data['phi'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.swpts, self.uml_source_data['swpts'], rtol=0.02, atol=1e-7)

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

    def test_single_update(self):
        self.tested.update(1)
        self.uml_source_data = loadmat('./test/singleUpdateGaussian.mat')['ans']
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.phi, [self.uml_source_data['phi']], rtol=0, atol=1e-7)

        # _psycfunc_derivative_mu
        inputs = [1, 1.50000000000000, 2, 2.50000000000000, 3, 3.50000000000000, 4, 4.50000000000000, 5, 5.50000000000000, 6, 6.50000000000000, 7, 7.50000000000000, 8, 8.50000000000000, 9, 9.50000000000000, 10, 10.5000000000000, 11, 11.5000000000000, 12, 12.5000000000000, 13, 13.5000000000000, 14, 14.5000000000000, 15, 15.5000000000000, 16, 16.5000000000000, 17, 17.5000000000000, 18, 18.5000000000000, 19, 19.5000000000000, 20]
        targets = [-0.0283130750429343, -0.0277988896555520, -0.0271197198135326, -0.0262881658497798, -0.0253193593585194, -0.0242305057247084, -0.0230403771281941, -0.0217687768406086, -0.0204359960563982, -0.0190622838489585, -0.0176673492040746, -0.0162699115981594, -0.0148873134448587, -0.0135352041462479, -0.0122273016826258, -0.0109752338828139, -0.00978845794157373, -0.00867425356832802, -0.00763778249820920, -0.00668220506395519, -0.00580884316012088, -0.00501737822959749, -0.00430607282690278, -0.00367200479095426, -0.00311130399602501, -0.00261938293268312, -0.00219115388464008, -0.00182122709928692, -0.00150408599568630, -0.00123423702523897, -0.00100633322676785, -0.000815271748844199, -0.000656266617096241, -0.000524898790644302, -0.000417146083272969, -0.000329395837871832, -0.000258443362838415, -0.000201479098307799, -0.000156067312838455]
        outputs = self.tested._psycfunc_derivative_mu(inputs)
        assert_allclose(outputs, targets, rtol=0, atol=1e-7)

        # _sigma2(_psycfunc_derivative_mu)
        targets = [227.883906465624, 226.712445804664, 227.737306267467, 231.010421050888, 236.663051576807, 244.915516532593, 256.093095899925, 270.649767373567, 289.202289180695, 312.578345713030, 341.884201221319, 378.599833974582, 424.713263555208, 482.911376885863, 556.852999185438, 651.562805493677, 774.004396166674, 933.921403771246, 1145.08318973896, 1427.14678740695, 1808.46602068769, 2330.36980300903, 3053.74040924077, 4069.22595289507, 5513.24944304471, 7593.35138395808, 10628.7053302042, 15115.5380191812, 21833.8266390619, 32023.0831971981, 47674.9210882742, 72025.0024699785, 110388.821586693, 171596.470528758, 270481.560807722, 432244.517128867, 700183.303405067, 1149537.29656271, 1912545.77027436]
        outputs = self.tested._sigma2(inputs, self.tested._psycfunc_derivative_mu)
        assert_allclose(outputs, targets, rtol=0, atol=1e-07)

        # _psycfunc_derivative_sigma
        targets = [-0.00534952308773938, -0.00747755348370201, -0.00946568217440189, -0.0112796968095015, -0.0128907081757469, -0.0142758937517798, -0.0154189884085470, -0.0163105101034430, -0.0169477220985849, -0.0173343462444087, -0.0174800534341263, -0.0173997668136666, -0.0171128202630248, -0.0166420187861340, -0.0160126487018010, -0.0152514840585749, -0.0143858318135121, -0.0134426524650380, -0.0124477855502716, -0.0114253012871794, -0.0103969912387537, -0.00938200273295789, -0.00839661434394231, -0.00745414338154855, -0.00656497128801557, -0.00573666922720207, -0.00497420398685349, -0.00428020351713045, -0.00365526184126604, -0.00309826448554282, -0.00260671774379485, -0.00217706776543282, -0.00180499839462321, -0.00148569967573960, -0.00121410179603551, -0.000985071821622885, -0.000793572801525130, -0.000634786612599496, -0.000504203278359704]
        outputs = self.tested._psycfunc_derivative_sigma(inputs)
        assert_allclose(outputs, targets, rtol=0, atol=1e-7)
        
        # _sigma2(_psycfunc_derivative_sigma)
        targets = [6383.48148880338, 3133.36656870776, 1869.39354856137, 1254.75025090565, 913.024290866890, 705.561124261336, 571.827810571935, 482.103878639024, 420.504529957629, 378.001756991518, 349.249913086106, 331.027405028031, 321.429187652552, 319.436870240446, 324.694899587195, 337.411314693148, 358.345560487705, 388.870155540426, 431.109325375962, 488.172060578721, 564.513455961173, 666.480451966500, 803.131101037971, 987.467474456065, 1238.30285361046, 1583.11281830154, 2062.42869542547, 2736.67344816897, 3696.90429332014, 5081.86714173441, 7105.35189130994, 10100.5302414861, 14592.5800240910, 21418.9135536320, 31930.3589763832, 48331.4601081855, 74262.3978303869, 115805.062533726, 183241.742322315]
        outputs = self.tested._sigma2(inputs, self.tested._psycfunc_derivative_sigma)
        assert_allclose(outputs, targets, rtol=0, atol=1e-7)
        # swpts
        assert_allclose(self.tested.swpts, [self.uml_source_data['swpts']], rtol=0, atol=1e-7)
        # p
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)
        # xnext
        # code.interact(local=locals())
        self.assertAlmostEquals(self.tested.xnext, self.uml_source_data['xnext'])

    def test_final_run(self):
        responses = self.uml_source_data['r']
        for response in responses:
            self.tested.update(response, incorrect_set=(0,))
        assert_allclose(self.tested.x, self.uml_source_data['x'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.phi, self.uml_source_data['phi'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.p, self.uml_source_data['p'], rtol=0, atol=1e-7)
        assert_allclose(self.tested.swpts, self.uml_source_data['swpts'], rtol=0.02, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
    
