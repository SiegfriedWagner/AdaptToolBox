from support import ABCAdaptation
from abc import ABC, abstractmethod
class Staircase(ABCAdaptation, ABC):
    default=""
    required_parameters=('ndown', 'nup', 'n', 'reset_after_change')

    def __init__(self, config=default, **kwargs):
        self.required_parameters += Staircase.required_parameters
        super(Staircase, self).__init__(config, **kwargs)

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


class LinearStarcase(Staircase):
    '''
    Required parameters:
    ndown
    nup
    n
    "diff_up"
    "diff_down"
    "value"
    "max_value"
    "min_value"
    '''
    default = ""
    required_parameters = ("diff_up", "diff_down", "value",
                           "max_value", "min_value")

    def __init__(self, config=default, **kwargs):
        self.required_parameters += LinearStarcase.required_parameters
        super(LinearStarcase, self).__init__(config, **kwargs)

    def up(self):
        self.value = min(self.value + self.diff_up, self.max_value)
        return self.value

    def down(self):
        self.value = max(self.value - self.diff_down, self.min_value)
        return self.value

    def no_change(self):
        return self.value
    
class ListStaircase(Staircase):
    default = ""
    required_parameters = ("stimuli_touple", "initial_position")

    def __init__(self, config=default, **kwargs):
        self.required_parameters += ListStaircase.required_parameters
        super(LinearStarcase, self).__init__(config, **kwargs)
        self.position = self.initial_position
        
    def up(self):
        self.position = max(self.position + 1,
                            len(self.stimuli_touple))
        return self.stimuli_touple[self.position]
    
    def down(self):
        self.position = min(self.position - 1, 0)
        return self.stimuli_touple[self.position]

    def no_change(self):
        return self.stimuli_touple[self.position]
    
if __name__ == '__main__':
    test = LinearStarcase(ndown=1, nup=2, n=2)
   
