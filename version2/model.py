import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class NeuralLayer:
    def __init__(self, input_dim: int, output_dim: int, name: str = ""):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Xavier初始化
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))
        
        # 保存梯度
        self.gradients = {
            'dW': np.zeros_like(self.W),
            'db': np.zeros_like(self.b)
        }
        
        # Adam優化器參數
        self.adam = {
            'mW': np.zeros_like(self.W),  # First moment
            'vW': np.zeros_like(self.W),  # Second moment
            'mb': np.zeros_like(self.b),
            'vb': np.zeros_like(self.b)
        }
        
    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        """前向傳播"""
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        return self.Z
    
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """反向傳播"""
        m = self.A_prev.shape[0]
        
        # 計算梯度
        self.gradients['dW'] = (1/m) * np.dot(self.A_prev.T, dZ)
        self.gradients['db'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # 梯度裁剪
        clip_value = 5.0
        self.gradients['dW'] = np.clip(self.gradients['dW'], -clip_value, clip_value)
        self.gradients['db'] = np.clip(self.gradients['db'], -clip_value, clip_value)
        
        # 計算上一層的梯度
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev
    
    def update_parameters(self, learning_rate: float, iteration: int,
                         beta1: float = 0.9, beta2: float = 0.999,
                         epsilon: float = 1e-8):
        """使用Adam優化器更新參數"""
        # 更新一階矩和二階矩
        self.adam['mW'] = beta1 * self.adam['mW'] + (1 - beta1) * self.gradients['dW']
        self.adam['vW'] = beta2 * self.adam['vW'] + (1 - beta2) * np.square(self.gradients['dW'])
        self.adam['mb'] = beta1 * self.adam['mb'] + (1 - beta1) * self.gradients['db']
        self.adam['vb'] = beta2 * self.adam['vb'] + (1 - beta2) * np.square(self.gradients['db'])
        
        # 偏差修正
        mW_corrected = self.adam['mW'] / (1 - beta1 ** iteration)
        vW_corrected = self.adam['vW'] / (1 - beta2 ** iteration)
        mb_corrected = self.adam['mb'] / (1 - beta1 ** iteration)
        vb_corrected = self.adam['vb'] / (1 - beta2 ** iteration)
        
        # 更新參數
        self.W -= learning_rate * mW_corrected / (np.sqrt(vW_corrected) + epsilon)
        self.b -= learning_rate * mb_corrected / (np.sqrt(vb_corrected) + epsilon)

class DetailedNeuralNetwork:
    def __init__(self, layer_dims: List[int], lambda_l2: float = 0.01):
        self.layers = []
        self.lambda_l2 = lambda_l2  # L2正則化係數
        
        # 創建層
        for l in range(1, len(layer_dims)):
            layer = NeuralLayer(layer_dims[l-1], layer_dims[l], f"Layer_{l}")
            self.layers.append(layer)
            
            # 除了最後一層外,都加入ReLU激活
            if l < len(layer_dims)-1:
                self.layers.append(ActivationLayer("relu"))
            else:
                self.layers.append(ActivationLayer("sigmoid"))
        
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def compute_cost(self, AL: np.ndarray, Y: np.ndarray, 
                    compute_regularization: bool = True) -> Tuple[float, float]:
        """
        計算損失（包含L2正則化）和準確率
        
        Args:
            AL: 模型輸出
            Y: 真實標籤
            compute_regularization: 是否計算正則化項
        """
        m = Y.shape[0]
        epsilon = 1e-15
        
        # 計算交叉熵損失
        AL = np.clip(AL, epsilon, 1 - epsilon)
        cross_entropy_cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        # 加入L2正則化
        l2_cost = 0
        if compute_regularization:
            for layer in self.layers:
                if isinstance(layer, NeuralLayer):
                    l2_cost += np.sum(np.square(layer.W))
            l2_cost = (self.lambda_l2 / (2 * m)) * l2_cost
        
        # 計算總損失
        cost = cross_entropy_cost + l2_cost
        
        # 計算準確率
        predictions = (AL > 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        
        return cost, accuracy
    
    def forward_propagation(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """前向傳播"""
        A = X
        
        for layer in self.layers:
            A = layer.forward(A, training)
        
        return A
    
    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray):
        """反向傳播"""
        m = Y.shape[0]

        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
        
        # 計算輸出層的梯度
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        dA = dAL
        for layer in reversed(self.layers):
            if isinstance(layer, ActivationLayer):
                dZ = layer.backward(dA)
                dA = dZ
            else:  # NeuralLayer
                dA = layer.backward(dA)
                # 加入L2正則化的梯度
                layer.gradients['dW'] += (self.lambda_l2 / m) * layer.W
    
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
             X_val: np.ndarray = None, Y_val: np.ndarray = None,
             learning_rate: float = 0.01,
             num_epochs: int = 100,
             batch_size: int = 32,
             print_cost: bool = True):
        """
        訓練神經網路
        
        Args:
            X_train: 訓練數據
            Y_train: 訓練標籤
            X_val: 驗證數據
            Y_val: 驗證標籤
            learning_rate: 學習率
            num_epochs: 訓練輪數
            batch_size: 批次大小
            print_cost: 是否打印成本
        """
        m = X_train.shape[0]
        iteration = 0
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            epoch_accuracy = 0
            num_batches = int(np.ceil(m / batch_size))
            
            # 打亂訓練數據
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]
            
            # 批次訓練
            for batch in range(num_batches):
                iteration += 1
                
                # 獲取當前批次
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # 前向傳播
                AL = self.forward_propagation(X_batch)
                
                # 計算成本和準確率
                batch_cost, batch_accuracy = self.compute_cost(AL, Y_batch)
                epoch_cost += batch_cost
                epoch_accuracy += batch_accuracy
                
                # 反向傳播
                self.backward_propagation(AL, Y_batch)
                
                # 更新參數
                current_lr = learning_rate / (1 + 0.01 * epoch)  # learning rate decay
                for layer in self.layers:
                    if isinstance(layer, NeuralLayer):
                        layer.update_parameters(current_lr, iteration)
            
            # 計算平均訓練指標
            epoch_cost /= num_batches
            epoch_accuracy /= num_batches
            self.metrics['train_loss'].append(epoch_cost)
            self.metrics['train_accuracy'].append(epoch_accuracy)
            
            # 如果有驗證集，計算驗證指標
            if X_val is not None and Y_val is not None:
                val_AL = self.forward_propagation(X_val, training=False)
                val_cost, val_accuracy = self.compute_cost(val_AL, Y_val)
                self.metrics['val_loss'].append(val_cost)
                self.metrics['val_accuracy'].append(val_accuracy)
                
                # Early stopping
                if val_cost < self.best_val_loss:
                    self.best_val_loss = val_cost
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break
                
                if print_cost and epoch % 10 == 0:
                    print(f"Epoch {epoch} | Train Loss: {epoch_cost:.4f} | "
                          f"Train Acc: {epoch_accuracy:.4f} | "
                          f"Val Loss: {val_cost:.4f} | Val Acc: {val_accuracy:.4f}")
            else:
                if print_cost and epoch % 10 == 0:
                    print(f"Epoch {epoch} | Train Loss: {epoch_cost:.4f} | "
                          f"Train Acc: {epoch_accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        Args:
            X: 輸入數據
        Returns:
            predictions: 預測結果
        """
        AL = self.forward_propagation(X, training=False)
        predictions = (AL > 0.5).astype(int)
        return predictions
    
    def plot_training_metrics(self):
        """繪製訓練指標"""
        plt.figure(figsize=(15, 5))
        
        # 繪製損失
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        if len(self.metrics['val_loss']) > 0:
            plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 繪製準確率
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_accuracy'], label='Train Accuracy')
        if len(self.metrics['val_accuracy']) > 0:
            plt.plot(self.metrics['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

class ActivationLayer:
    def __init__(self, activation_type: str = "relu", alpha: float = 0.01):
        self.type = activation_type
        self.alpha = alpha
        self.cache = {'Z': None, 'A': None}
    
    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:  # 加入training參數
        """
        前向傳播
        Args:
            Z: 輸入值
            training: 是否在訓練模式
        Returns:
            A: 激活值
        """
        self.cache['Z'] = Z
        
        if self.type == "relu":
            A = np.maximum(0, Z)
        elif self.type == "leaky_sigmoid":
            A = self.leaky_sigmoid(Z)
        else:  # sigmoid
            A = 1 / (1 + np.exp(-Z))
        
        self.cache['A'] = A
        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        反向傳播
        Args:
            dA: 激活值的梯度
        Returns:
            dZ: 輸入值的梯度
        """
        if self.type == "relu":
            dZ = dA * (self.cache['Z'] > 0)
        elif self.type == "leaky_sigmoid":
            dZ = dA * self.leaky_sigmoid_derivative(self.cache['Z'])
        else:  # sigmoid
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        
        return dZ
    
from prepare_data import prepare_data

# 讀取和準備數據
df = pd.read_csv('boxing-matches.csv')
X_train, X_test, y_train, y_test, scaler = prepare_data(df)

# 創建模型
model = DetailedNeuralNetwork(
    layer_dims=[X_train.shape[1], 16, 1],
    lambda_l2=0.00001
)

# 訓練模型
model.train(
    X_train, np.array(y_train).reshape(-1, 1),
    X_val=X_test, Y_val=np.array(y_test).reshape(-1, 1),
    learning_rate=0.0001,
    num_epochs=100,
    batch_size=32
)

# 評估模型
model.plot_training_metrics()