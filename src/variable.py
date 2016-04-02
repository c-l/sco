
class Variable(object):
    """
    Variable

    Manages Gurobi variables by maintaining an ordering of Gurobi variables,
    """

    def __init__(self, grb_vars):
        """
        _grb_vars: Numpy array of Gurobi variables. The ordering of the Gurobi
        variables must be maintained.
        _value: current value of this variable
        _saved_value: saved value of this variable
        """
        self._grb_vars = grb_vars
        self._value = None
        self._saved_value = None

    def get_grb_vars():
        return self._grb_vars.copy()

    def get_value():
        return self._value.copy()

    def add_trust_region(self, trust_box_size):
        """
        Adds a trust region around the current value by changing the upper and
        lower bounds of the Gurobi variables self._grb_vars
        """
        raise NotImplementedError

    def update(self):
        """
        If the gurobi variables have valid values, update self._value to reflect
        the values in the gurobi variables.

        When the gurobi variables do not have valid values, self._value is set
        to None
        """
        raise NotImplementedError

    def save(self):
        """
        Save the current value.

        TODO: remember to unit test that by modifying value you can't change the
        value in self
        """
        self._saved_value = self._value.copy()

    def restore(self):
        """
        Restore value to the saved value.
        """
        self._value = self._saved_value.copy()
