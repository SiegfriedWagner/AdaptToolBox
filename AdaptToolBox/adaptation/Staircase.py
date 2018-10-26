'''
Module containing staircase paradigm based algorithms.

Provides:
_Staircase -- abstract base class for all staircase class
LinearStaircase -- changes stimulus linearly based on participant
responses.
ListStaircase -- iterates over provided tuple based on participant
responses.
'''
from AdaptToolBox.support import ABCAdaptation
from abc import ABC, abstractmethod
class _Staircase(ABCAdaptation, ABC):
    '''
    Abstract base class for all staricase base algorithms
    
    Args required to include during child class construction:
    ndown: int -- number of consecutive correct responses
    nup: int -- number of consecutive incorrect resposes
    required to change stimuli up 
    n: int -- initial number of consecutive correct responses (n>0)
    reset_after_change: bool -- decides if reset n to 0 after every 
    stimulus change
    '''
    
    required_parameters = {'ndown': int,
                           'nup': int,
                           'n': int,
                           'reset_after_change': bool}

    def __init__(self, **kwargs):
        super(_Staircase, self).__init__(**kwargs)

    @ABCAdaptation.answer_boolcheck()  # That brackets are important
    def update(self, response):
        self.n = 0 if (self.n*response) < 0 else self.n
        if self.n + response == self.ndown:
            self.n = 0 if self.reset_after_change else self.n
            return self.down()
        elif self.n + response == (-self.nup):
            self.n = 0 if self.reset_after_change else self.n
            return self.up()
        else:
            self.n = max(0, self.n*response)*response + response
            return self.no_change()

    @abstractmethod
    def up(self):
        return NotImplementedError

    @abstractmethod
    def down(self):
        raise NotImplementedError

    @abstractmethod
    def no_change(self):
        raise NotImplementedError


class LinearStaircase(_Staircase):
    '''
    Algorithms that changes stimulus linearly based on participant
    responses.

    Args:
    ndown: int -- number of consecutive correct responses
    required to change stimuli down
    nup: int -- number of consecutive incorrect resposes
    required to change stimuli up 
    n: int -- initial number of consecutive correct responses (n>0)
    or incorrect responses (n<0)
    diff_up: flaot -- the amount by which stimulus changes after nup
    cosecutive correct responses
    diff_down: float -- the amount by which stimulus changes after ndown
    cosecutive incorrect responses
    value: float -- initial value of stimulus
    max_value: float -- maximum value of stimulus
    min_value: float -- minimum value of stimulus
    reset_after_change: bool -- decides if reset n to 0 after every 
    stimulus change
    '''

    required_parameters = {"diff_up": float,
                           "diff_down": float,
                           "value": float,
                           "max_value": float,
                           "min_value": float}

    def __init__(self, **kwargs):
        super(LinearStaircase, self).__init__(**kwargs)

    def up(self):
        self.value = min(self.value + self.diff_up, self.max_value)
        return self.value

    def down(self):
        self.value = max(self.value - self.diff_down, self.min_value)
        return self.value

    def no_change(self):
        return self.value
    
class ListStaircase(_Staircase):
    '''
    Staircase that based on response returns items from tuple, where
    correct responses move iterator up and incorrent move it down.

    Args:
    ndown: int -- number of consecutive correct responses
    required to change stimuli down
    nup: int -- number of consecutive incorrect resposes
    required to change stimuli up 
    n: int -- initial number of consecutive correct responses (n>0)
    or incorrect responses (n<0)
    stimuli_tuple: tuple -- tuple contating all possible stimuli
    initial_position: int -- starting position in toupe
    reset_after_change: bool -- decides if reset n to 0 after every 
    stimulus change
    '''

    required_parameters = {"stimuli_tuple": tuple,
                           "initial_position": int}

    def __init__(self, **kwargs):
        super(ListStaircase, self).__init__(**kwargs)
        self.position = self.initial_position
        
    def up(self):
        self.position = min(self.position + 1,
                            len(self.stimuli_tuple)-1)
        return self.stimuli_tuple[self.position]
    
    def down(self):
        self.position = max(self.position - 1, 0)
        return self.stimuli_tuple[self.position]

    def no_change(self):
        return self.stimuli_tuple[self.position]
  
