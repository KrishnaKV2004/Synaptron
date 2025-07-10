from losses import mse_loss, mse_loss_derivative

class Sequential:

    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dLoss):
        grad = dLoss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)
    
    def fit(self, X, y, epochs=10, learning_rate=0.01, verbose=True):
        
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = mse_loss(y, y_pred)
            dLoss = mse_loss_derivative(y, y_pred)
            self.backward(dLoss)
            self.update_params(learning_rate)
            
            if verbose:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        return self.forward(X)