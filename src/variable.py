
class Variable(object):

    def __init__(self, grb_vars):
        self._grb_vars = grb_vars
        self._value = None
        self._saved_value = None

    def get_grb_vars():
        return self._grb_vars.copy()

    def get_value():
        return self._value.copy()

    def update(self):
        """
        If the gurobi variables have valid values, update self._value to reflect
        the values in the gurobi variables.
        """
        raise NotImplementedError

    def save(self):
        """
        Save the current value.
        """
        self._saved_value = self._value.copy()

    def restore(self):
        """
        Restore value to the saved value.
        """
        self._value = self._saved_value.copy()
