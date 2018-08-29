import AdaptToolBox.support as support
import unittest

class TestABCAdaptation(unittest.TestCase):
    class StubABCAdaptation(support.ABCAdaptation):
        def __init__(self):
            pass
        
        def update(self):
            pass

    def test_validate_input(self):
        required = {
            "float": float,
            "int": int,
            "list": ["valid_1", 1.2, 1],
            "dict": {"one": int,
                     "two": int}
        }

        inputs = {"float": 1.2,
                  "int": 1,
                  "list": "not_valid",
                  "dict": {"one": 1,
                           "two": 2}
            
        }
        with self.assertRaises(ValueError):
            support.ABCAdaptation.validate_input(required, inputs)

        inputs = {
            "float": 1.2,
            "int": 1.2,
            "list": 1.2,
            "dict": {"one": 1,
                     "two": 2}
        }
        with self.assertRaises(ValueError):
            support.ABCAdaptation.validate_input(required, inputs)
            
if __name__ == '__main__':
    unittest.main()
