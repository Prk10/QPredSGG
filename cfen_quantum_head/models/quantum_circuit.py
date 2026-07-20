import pennylane as qml
import torch.nn as nn
import torch

import pennylane as qml
import torch.nn as nn
import torch

class QuantumLayer(nn.Module):
    """
    Multi-Head Quantum Circuit Layer.
    Configured for 4 qubits, 4 heads, and 2 layers (96 parameters).
    """
    def __init__(self, n_qubits=4, n_layers=2, n_heads=4, q_device=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_qubits = n_qubits 
        
        # 1. Dynamic Device Assignment
        if q_device is None:
            # KAGGLE DRY RUN MODE
            self.dev = qml.device("default.qubit", wires=n_qubits)
            diff_strategy = "backprop" 
        else:
            # IBM HARDWARE MODE
            self.dev = q_device
            diff_strategy = "parameter-shift" 

        # 2. Encapsulated Quantum Node
        @qml.qnode(self.dev, interface="torch", diff_method=diff_strategy)
        def qnode(inputs, weights):
            # Compress classical features into probability amplitudes
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_qubits), normalize=True)
            
            # Apply parameterized entanglement (weights shape: n_layers, n_qubits, 3)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Measure the expectation value of the PauliZ observable on all wires
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # 3. Parallel TorchLayer Initialization
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        self.heads = nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_heads)
        ])
        
    def forward(self, x):
        """
        Input x: [Batch, 16] (4 heads * 4 qubits)
        """
        chunks = torch.chunk(x, self.n_heads, dim=1)
        outputs = []
        
        for i, layer in enumerate(self.heads):
            out = layer(chunks[i]) 
            outputs.append(out)
            
        return torch.cat(outputs, dim=1)

'''
n_qubits = 16 


dev = qml.device("lightning.gpu", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qnode(inputs, weights):
    
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    
    
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    
    def __init__(self, n_layers=4, n_heads=1): 
        super().__init__()
        self.n_heads = n_heads
        self.n_qubits = n_qubits 
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        self.heads = nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_heads)
        ])
        
    def forward(self, x):
        chunks = torch.chunk(x, self.n_heads, dim=1)
        outputs = []
        for i, layer in enumerate(self.heads):
            out = layer(chunks[i]) 
            outputs.append(out)
        return torch.cat(outputs, dim=1)

'''

'''
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_layers=4, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_qubits = n_qubits 
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # 1. Initialize TorchLayers
        layers = [qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_heads)]
        self.heads = nn.ModuleList(layers)
        
    def forward(self, x):
        # x is already on the GPU from your classical layers
        chunks = torch.chunk(x, self.n_heads, dim=1)
        outputs = []
        for i, layer in enumerate(self.heads):
            # Because 'chunks' is on GPU, and the TorchLayer weights are registered 
            # to your main model (which you put on GPU), PennyLane will do the 
            # math using PyTorch CUDA operations!
            out = layer(chunks[i]) 
            outputs.append(out)
            
        return torch.cat(outputs, dim=1)

'''
'''
n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    #qml.AngleEmbedding(inputs, wires=range(n_qubits))

    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

    #1st Quantum Experiment:
    #qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    #2nd Quantum Experiment:
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] #returns a smooth output to pass to the classical layer

#Multi Head Quantum approach
#4 circuits x 4 qubits
class QuantumLayer(nn.Module):
    #1st Quantum Experiment:
    #n_layers=2, n_heads=1

    #2nd Quantum Experiment
    #n_layers = 4. n_heads=32
    def __init__(self, n_layers=2, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.n_qubits = n_qubits # 4
        
        #1st Quantum Experiment
        #weight_shapes = {"weights": (n_layers, n_qubits)}

        #2nd Quantum Experiment
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Create a list of quantum layers (heads)
        # We use ModuleList so PyTorch registers their parameters correctly
        self.heads = nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_heads)
        ])
        
    def forward(self, x):
        """
        Input x: [Batch, n_heads * n_qubits]
        """
        # Split input into chunks for each head
        # x shape: [Batch, 16] -> chunks of [Batch, 4]
        chunks = torch.chunk(x, self.n_heads, dim=1)
        
        # Process each chunk through its respective quantum circuit
        outputs = []
        for i, layer in enumerate(self.heads):
            out = layer(chunks[i]) # [Batch, 4]
            outputs.append(out)
            
        # Concatenate results back together
        # Result shape: [Batch, n_heads * 4]
        return torch.cat(outputs, dim=1)
'''
"""
class QuantumLayer(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()

        weight_shapes = {"weights": (n_layers, n_qubits)}

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)
"""
