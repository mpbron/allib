class NoVectorsException(Exception):
    """This exception is thrown when there are no vectors available
    """    
    pass

class NoLabeledDataException(Exception):
    """This exception is thrown when there are no labeled instances available
    """    
    pass

class NotInitializedException(Exception):
    """This exception is returned if the Active Learner has not been 
    initialized. That is, there is no attached Environment, so it 
    cannot sample instances.
    """    
    pass

class NoOrderingException(Exception):
    """This exception is returned if the instances in the `ActiveLearner`
    have not yet been ordered, or establishing an ordering is not possible
    while sampling instances. In this case, no instances can be returned
    and instead this `Exception` is raised.
    """    
    pass
