'''
Code used in various parts of package.
'''
from abc import ABC, abstractmethod
import sys
import pdb
import functools
import traceback
import pickle

def debug_on(*exceptions):
    if not exceptions:
        exceptions = (AssertionError, )
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info) 
                pdb.post_mortem(info[2])
        return wrapper
    return decorator
'''Loadmat source: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries'''
import scipy.io as spio
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
        return dict        

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''


         

        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class ABCAdaptation(ABC):
    '''
    Abstrac Base Class for every adaptation algorithm in this package.
    

    Methods:
    __init__ (abstractmethod)
    update (abstractmethod)
    answer_boolcheck (staticmethod, decorator)
    '''
    required_parameters = {}

    @abstractmethod
    def __init__(self, **kwargs):
        '''Takes care of intialization and declaration of arguments'''
        self.validate_input(self.__class__.required_parameters, kwargs)
        for key, value in kwargs.items():
            if key in self.__class__.required_parameters:
                setattr(self, key, value)
            else:
                raise AttributeError(str(key) + " is not used by "
                                     + str(self.__class__.__name__))
        missing = set(self.__class__.required_parameters).difference(dir(self))
        if len(missing) != 0:
            raise AttributeError("Missing parameters " + str(missing))

    def __init_subclass__(cls):
        '''
        Takes care of properly updating "required_parameters"
        variable in subclasses
        '''
        if hasattr(cls, "required_parameters"):
            for mr in cls.mro():
                if issubclass(cls, mr) and hasattr(mr, "required_parameters"):
                    cls.required_parameters.update(mr.required_parameters)
                
    @staticmethod
    def answer_boolcheck(correct_set: set = set((1.0, "correct", "valid")),
                         incorrect_set: set = set((-1.0, "incorrect", "missed"))):
        '''
        Function decorator that checks if given answer belongs to 
        either correct_set variable or incorrect_set.
        
        Keyword arguments:
        correct_set: set -- arguments that makes decorator pass 1
        incorrect_set: set -- arguments that makes decorator pass -1
        '''
        def real_decorator(func):
            def func_wraper(obj, response,
                            correct_set=correct_set,
                            incorrect_set=incorrect_set):
                if type(response) is str:
                    response = str.lower(response)
                if response in correct_set:
                    response = 1
                elif response in incorrect_set:
                    response = -1
                else:
                    raise ValueError(str(response) +
                                     " is not in any validated response list")
                return func(obj, response)
            return func_wraper
        return real_decorator

    @abstractmethod
    def update(self, response: int):
        '''
        Function that takes response defined as
        1 = correct
        -1 = incorrect
        and generates new stimulus based on it.
        '''
        raise NotImplementedError

    @staticmethod
    def validate_input(required, inputs, verbose=False):
        for key, value in inputs.items():
            if key in required.keys():
                valid = required[key]
                if type(valid) is type and valid is type(value):
                    pass
                elif type(valid) is list and value in valid:
                    pass
                elif type(valid) is dict and type(value) is dict:
                    ABCAdaptation.validate_input(valid, value, verbose)
                else:
                    raise ValueError(str(key) + " is not " + str(valid))
                if verbose: print("{} have valid value of type {}".format(key, value))
            else:
                raise ValueError("Unknown input: " + str(key))
            
    def save_as_pickle(self, file_path):
        pickle.dump(self, open(file_path, 'wb'))

    def __repr__(self):
        represent = str(self.__class__.__name__) + '('
        for key in self.__class__.required_parameters.keys():
            value = getattr(self, key)
            if type(value) is str:
                represent += '\n\t' + key + '=' + '"' + value + '"' + ','
            else:
                represent += '\n\t' + key + '=' +str(value) + ','
        represent = represent[:-1] + ')'
        return represent
    
class InheritableDocstrings(ABC):
    def __prepare__(name, bases):
        classdict = dict()

        # Construct temporary dummy class to figure out MRO
        mro = type('K', bases, {}).__mro__[1:]
        assert mro[-1] == object
        mro = mro[:-1]

        def inherit_docstring(fn):
            if fn.__doc__ is not None:
                raise RuntimeError('Function already has docstring')

            # Search for docstring in superclass
            for cls in mro:
                super_fn = getattr(cls, fn.__name__, None)
                if super_fn is None:
                    continue
                fn.__doc__ = super_fn.__doc__
                break
            else:
                raise RuntimeError("Can't inherit docstring for %s: method does not "
                                   "exist in superclass" % fn.__name__)

            return fn

        classdict['inherit_docstring'] = inherit_docstring
        return classdict
