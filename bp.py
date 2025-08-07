import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    

class Linear(Module):
    def __init__(self,  in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = np.random.randn(in_dim,  out_dim)
        self.grad_w = np.zeros_like(self.w)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.w  

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_w = self.input.T @ grad 
        return grad @ self.w.T


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask  


class Network(Module):
    def __init__(self, n_layer: int ,dim: list[int]) -> None:
        """ 
        here dim includes the input dim for simplicity, so its n_layer + 1
        """
        super().__init__()
        assert n_layer == len(dim) - 1

        self.layers: list[Module] = []
        for i in range(n_layer):
            self.layers.append(Linear(dim[i], dim[i+1]))
            if i < n_layer - 1:  # Don't add ReLU after the last layer
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update_weights(self, lr: float) -> None:
        """Update weights using gradient descent"""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.w -= lr * layer.grad_w
    
    def zero_grad(self) -> None:
        """Zero all gradients"""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.grad_w.fill(0)


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Mean Squared Error loss
    Returns: (loss_value, gradient)
    """
    diff = pred - target
    loss = np.mean(diff ** 2)
    grad = 2 * diff / len(pred)  # derivative of MSE
    return loss, grad


if __name__ == '__main__':
    lr = 1e-3  
    hidden_size = 12 
    samples = 1000
    label_size = 1
    epoch = 1000  

    dim = [hidden_size] +  [24, 48] + [label_size]
    n_layer = len(dim) - 1
    model = Network(n_layer, dim)
    data = np.random.randn(samples, dim[0])
    label = np.random.randn(samples, label_size)

    for i in range(epoch):
        pred = model(data)
        loss, loss_grad = mse_loss(pred, label)
        model.zero_grad()
        model.backward(loss_grad)
        model.update_weights(lr)
        if i % 10 == 0:    
            print(f"Epoch {i}, Loss: {loss:.8f}")