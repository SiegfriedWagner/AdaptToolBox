from support import ABCAdaptation, InheritableDocstrings
from adaptation.Staircase import LinearStarcase
import curves
import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scistat
import scipy.special as scispec
import math
from abc import ABC, abstractmethod
import code
import inspect

class UMLParameter(object):
    '''
    UMLParameter used in UML.

    Args:
    value: float -- constant value of parameter
        
    min_value: float -- minimum value of parameter
    max_value: float -- maximum value of parameter
    n : int -- number of values generated between min and max value
    scale: str -- method used to create parameter space between min and max value,
    currently supported methods are 'lin' and 'log'
    dist: str -- method used to calculate prior propability of every value,
    currently supported methods are 'norm' and 'flat'
    mu: flaot -- mean value used to calculated propability density function
    std: float -- standard deviation used to calculate propability density function

    Called function requires to set either:
    1. ONLY value parameter
    or
    2. all parameters EXCEPT value
    '''
   
    def __init__(self,
                 value: float = None,
                 min_value: float = None,
                 max_value: float = None,
                 scale: str = None,
                 dist: str = None,
                 n: int = None,
                 mu: float = None,
                 std: float = None) -> None:
        local = locals()
        copy = local.copy()
        del copy['value']
        del copy['self']
        none_test = list(map(lambda x: True if x is None else False, list(copy.values())))

        if local['value'] is not None and all(none_test):
            self.value = value
        elif local['value'] is None and not any(none_test):
            self.min_value = min_value
            self.max_value = max_value
            self.scale = scale
            self.dist = dist
            self.n = n
            self.mu = mu
            self.std = std
        else:
            raise ValueError("Invalid arguments configuration. ")
    def setParSpace(self):
        if hasattr(self, "value"):
            self.space = self.value
        else:
            if self.scale == 'lin':
                self.space = np.linspace(self.min_value,
                                         self.max_value,
                                         self.n, dtype=np.float64)
            elif self.scale == 'log':
                self.space = np.geomspace(self.min_value,
                                          self.max_value,
                                          self.n, dtype=np.float64)
            else:
                raise Exception('Variable have illegal scale name '
                                + str(self.scale))
    
    def setPrior(self):
        if self.dist == 'norm':
            if self.scale == 'lin':
                return scistat.norm.pdf(self.meshgrid, self.mu, self.std)
            elif self.scale == 'log':
                return scistat.norm.pdf(np.log10(self.meshgrid), self.mu, self.std)
            else:
                raise ValueError('Variable have illegal scale name '
                                 + str(self.dist))
        elif self.dist == 'flat':
            return np.ones_like(self.meshgrid)

class UML(ABCAdaptation, ABC):
    '''
    Abstrac Base Class for all UML Classes
    '''
    default = ""
    required_parameters = ("max_stimuli", "min_stimuli", "value",
                           "method", "alpha", "beta", "gamma", "lamb")

    def __init__(self, config=default, **kwargs):
        self.required_parameters += UML.required_parameters
        super(UML, self).__init__(config, **kwargs)
        self._setP0()
        self.x = []
        self.trial_n = 0
        self.responses = []
        self.step_list = []
        self.swpts = np.empty((0, 4))
        self.swpts_idx = []

        if self.alpha.n > 1:
            self.swpts_idx.append(1)
        if self.beta.n > 1:
            self.swpts_idx.append(0)
            self.swpts_idx.append(2)
        if self.lamb.n > 1:
            self.swpts_idx.append(3)
        self.swpts_idx.sort()
        self.phi = np.empty((0, 4))
        self.xnext = self.value
        if self.xnext == self.min_stimuli:
            init_step = 0
        elif self.xnext == self.max_stimuli:
            init_step = len(self.swpts_idx) - 1 
        else:
            init_step = math.ceil((len(self.swpts_idx) - 1) / 2)
        self.swpt_picker = LinearStarcase(safemode=False,
                                          reset_after_change=True,
                                          ndown=2,
                                          nup=1,
                                          n=0,
                                          diff_up=1,
                                          diff_down=1,
                                          value=init_step,
                                          max_value=len(self.swpts_idx)-1,
                                          min_value=0)


    def _setP0(self):

        # set the space for phi
        self.alpha.setParSpace()
        self.beta.setParSpace()
        self.lamb.setParSpace()
        # code.interact(local=locals())
        self.alpha.meshgrid, self.beta.meshgrid, self.lamb.meshgrid = np.meshgrid(self.alpha.space,
                                                                            self.beta.space,
                                                                            self.lamb.space)
        # set the prior value and the space for hypo phi
        A = self.alpha.setPrior()
        B = self.beta.setPrior()
        L = self.lamb.setPrior()
        self.p = np.log(self._prepare_prob(A * B * L))

    @staticmethod
    def _prepare_prob(x):
        x = x * (1 - 1e-10)
        x = x / np.sum(x)
        return x
    
    @ABCAdaptation.answer_boolcheck()  # That brackets are important
    def update(self, answer) -> float:
        '''
        Generates new stimuli based on answer.

        Args:
        answer -- discrete response in any supported format.
        By default only supported values are '1' for correct
        answer and '-1' for incorrect answer.

        Optional args:
        correct_set: set -- set of values converted to correct response
        incorrect_set: set -- set of values converted to incorrect response

        Returns:
        float -- new stimulus to procede 
        '''
        self.x.append(self.xnext)
        self.responses.append(answer)
        self.p = self.p + \
                 np.log(self._prepare_prob(
                     self._prob_function(self.xnext, self.alpha.meshgrid,
                                        self.beta.meshgrid, self.gamma.value,
                                        self.lamb.meshgrid)) ** max(0, answer)) + \
                 np.log(
                     self._prepare_prob(
                         1 - self._prob_function(self.xnext,
                                                self.alpha.meshgrid,
                                                self.beta.meshgrid,
                                                self.gamma.value,
                                                self.lamb.meshgrid)) ** (1 - max(0, answer)))
        self.p = self.p - np.nanmax(self.p)

        if self.method == 'mode':
            idx = np.argmax(self.p)
            self.phi = np.append(self.phi, [[self.alpha.item(idx),
                                             self.beta.item(idx),
                                             self.gamma,
                                             self.lamb.item(idx)]], axis=0)
        elif self.method == 'mean':
            pdf_tmp = np.exp(self.p)
            pdf_tmp = pdf_tmp / np.sum(pdf_tmp)
            # alpha
            alpha_est_tmp = np.sum(pdf_tmp * self.alpha.meshgrid)
            # beta
            beta_est_tmp = np.sum(pdf_tmp * self.beta.meshgrid)
            # lambda
            lambda_est_tmp = np.sum(pdf_tmp * self.lamb.meshgrid)
            # combine together
            self.phi = np.append(self.phi, [[alpha_est_tmp, beta_est_tmp, self.gamma.value, lambda_est_tmp]], axis=0)
        else:
            raise Exception('Wrong method name')
        # find the next signal strength at the appropriate sweet point
        swpt = np.clip(self._calc_sweetpoints(self.phi[-1]), self.min_stimuli, self.max_stimuli)
        swpt = np.append(swpt, [self.max_stimuli])
        # staircase algorithm chooses sweet points
        self.xnext = swpt[self.swpt_picker.update(answer)]
        self.swpts = np.append(self.swpts, [swpt], axis=0)
        self.trial_n += 1

        return self.xnext

    @abstractmethod
    def _calc_sweetpoints(self, phi):
        raise NotImplementedError

    @abstractmethod
    def _prob_function(self, x, alpha, beta, gamma, lamb):
        raise NotImplementedError


class LogitUML(UML, InheritableDocstrings):
    '''
    UML using logistic function.

    Args:
    safemode: bool -- currently unused, set False
    max_stimuli: float -- maximum stimulus generated by algorithm
    min_stimuli: float -- minimum stimulus generated by algorithm
    value: float -- initial value of stimulus
    method: str -- method used to pick sweetpoints form porability space
    currentyly supported 'mode' and 'mean'
    alpha: UMLParameter -- alpha parameter
    beta: UMLParameter -- beta parameter
    gamma: UMLParameter -- gamma parameter
    lamb: UMLParameter -- lambda parameter
    '''
    def _prob_function(self, x, alpha, beta, gamma, lamb):
        return curves.logistic(x, alpha, beta, gamma, lamb)
        
    def _calc_sweetpoints(self, phi):

        def alphavar_est(x, alpha, beta, gamma, lamb):
            term1 = np.exp(2 * beta * (alpha - x))
            term2 = (1 + np.exp(beta * (x - alpha))) ** 2
            term3 = -gamma + (lamb - 1) * np.exp(beta * (x - alpha))
            term4 = 1 - gamma + lamb * np.exp(beta * (x - alpha))
            term5 = beta ** 2 * (-1 + gamma + lamb) ** 2

            return -term1 * term2 * term3 * term4 / term5

        def betavar_est(x, alpha, beta, gamma, lamb, switch):

            term1 = np.exp(2 * beta * (alpha - x))
            term2 = (1 + np.exp(beta * (x - alpha))) ** 2
            term3 = -gamma + (lamb - 1) * np.exp(beta * (x - alpha))
            term4 = 1 - gamma + lamb * np.exp(beta * (x - alpha))
            term5 = (x - alpha) ** 2 * (-1 + gamma + lamb) ** 2
            try:
                value = -term1 * term2 * term3 * term4 / term5
            except ZeroDivisionError:
                return float('Inf')
            if (x >= alpha and switch == -1) or (x <= alpha and switch == 1):

                return value + 1e10
            else:
                return value

        # calculate the sweet points for a logit psychometric function, for parameter alpha, beta, and lambda
        alpha = phi[-4]
        beta = phi[-3]
        gamma = phi[-2]
        lamb = phi[-1]
        swpts = np.zeros(3)
        switch = -1
        swpts[0] = sciopt.fmin(betavar_est, x0=alpha - 10.0, args=(alpha, beta, gamma, lamb, switch), disp=0)[0]
        switch = 1
        swpts[2] = sciopt.fmin(betavar_est, x0=alpha + 10.0, args=(alpha, beta, gamma, lamb, switch), disp=0)[0]
        swpts[1] = sciopt.fmin(alphavar_est, alpha, args=(alpha, beta, gamma, lamb), disp=0)[0]
        swpts = sorted(swpts)

        return swpts

class WeibullUML(UML):
    '''
    UML using weibull function.

    Args:
    safemode: bool -- currently unused, set False
    max_stimuli: float -- maximum stimulus generated by algorithm
    min_stimuli: float -- minimum stimulus generated by algorithm
    value: float -- initial value of stimulus
    method: str -- method used to pick sweetpoints form porability space
    currentyly supported 'mode' and 'mean'
    alpha: UMLParameter -- alpha parameter
    beta: UMLParameter -- beta parameter
    gamma: UMLParameter -- gamma parameter
    lamb: UMLParameter -- lambda parameter
    '''

    def _prob_function(self, x, alpha, beta, gamma, lamb):
        return curves.weibull(x, alpha, beta, gamma, lamb)
    
    def _calc_sweetpoints(self, phi):
        def kvar_est(x, k, beta, gamma, lamb):
            term1 = k ** 2 * (x / k) ** (-2 * beta)
            term2 = -1 + gamma - np.exp((x / k) ** beta) * (-1 + lamb) + lamb
            term3 = -1 + gamma + lamb - np.exp((x / k) ** beta) * lamb
            term4 = beta ** 2 * (-1 + gamma + lamb) ** 2
            try:
                value = -term1 * term2 * term3 / term4
            except ZeroDivisionError:
                value = float('Inf')
            return value

        def betavar_est(x, k, beta, gamma, lamb, switch):
            term1 = (x / k) ** (-2 * beta)
            term2 = -1 + gamma - np.exp((x / k) ** beta) * (-1 + lamb) + lamb
            term3 = -1 + gamma + lamb - np.exp((x / k) ** beta) * lamb
            term4 = (-1 + gamma + lamb) ** 2 * np.log(x / k) ** 2
            
            if (x >= k and switch == -1) or (x <= k and switch == 1):
                return  -term1 * term2 * term3 / term4 + 1e10
            else:
                return -term1 * term2 * term3 / term4

        k = phi[-4]
        beta = phi[-3]
        gamma = phi[-2]
        lamb = phi[-1]
        swpts = [0, 0, 0]
        switch = -1
        swpts[0] = sciopt.fmin(betavar_est, k / 2, args=(k, beta, gamma, lamb, switch), disp=0)[0]
        switch = 1
        swpts[2] = sciopt.fmin(betavar_est, k * 2, args=(k, beta, gamma, lamb, switch), disp=0)[0]
        swpts[1] = sciopt.fmin(kvar_est, k, args=(k, beta, gamma, lamb), disp=0)[0]
        swpts = sorted(swpts)

        return swpts
