
class ContinuationNotDefinedException(Exception):

    def __init__(self):
        self.message = "Continuation method must be string either 'v' for continuation\
             on velocity or 'r' for continuation on receivers"
        
        super().__init__(self.message)
