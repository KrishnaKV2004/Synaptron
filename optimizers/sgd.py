class SGD:
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def step(self, model):
        model.update_params(self.learning_rate)