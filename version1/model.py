import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralLayer:
    def __init__(self, input_dim: int, output_dim: int, name: str = ""):
        """
        初始化神經網路層
        Args:
            input_dim: 輸入維度
            output_dim: 輸出維度
            name: 層的名稱
        """
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 初始化權重矩陣 W (input_dim x output_dim)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        
        # 初始化偏置向量 b (1 x output_dim)
        self.b = np.zeros((1, output_dim))
        
        # 存儲訓練過程中的數值
        self.cache: Dict = {
            'A_prev': None,  # 上一層的激活值 (m x input_dim)
            'Z': None,       # 當前層的線性輸出 (m x output_dim)
            'A': None        # 當前層的激活輸出 (m x output_dim)
        }
        
        # 存儲梯度
        self.gradients: Dict = {
            'dW': None,  # 權重的梯度 (input_dim x output_dim)
            'db': None,  # 偏置的梯度 (1 x output_dim)
            'dA': None,  # 激活值的梯度 (m x output_dim)
            'dZ': None   # 線性輸出的梯度 (m x output_dim)
        }
        
        # 存儲參數更新量
        self.momentum: Dict = {
            'vW': np.zeros_like(self.W),  # 權重的動量
            'vb': np.zeros_like(self.b)    # 偏置的動量
        }
    
    def linear_forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        前向傳播的線性部分 Z = W·A + b
        Args:
            A_prev: 上一層的激活值 (m x input_dim)
        Returns:
            Z: 當前層的線性輸出 (m x output_dim)
        """
        self.cache['A_prev'] = A_prev
        Z = np.dot(A_prev, self.W) + self.b  # 矩陣乘法 + 廣播
        self.cache['Z'] = Z
        return Z
    
    def linear_backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        反向傳播的線性部分
        Args:
            dZ: 當前層線性輸出的梯度 (m x output_dim)
        Returns:
            dA_prev: 上一層激活值的梯度 (m x input_dim)
        """
        m = self.cache['A_prev'].shape[0]
        
        # 計算權重的梯度
        self.gradients['dW'] = (1/m) * np.dot(self.cache['A_prev'].T, dZ)
        # 計算偏置的梯度
        self.gradients['db'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        # 計算上一層激活值的梯度
        dA_prev = np.dot(dZ, self.W.T)
        
        return dA_prev
    
    def update_parameters(self, learning_rate: float, beta: float = 0.9):
        """
        使用動量更新參數
        Args:
            learning_rate: 學習率
            beta: 動量係數
        """
        # 更新動量
        self.momentum['vW'] = beta * self.momentum['vW'] + (1 - beta) * self.gradients['dW']
        self.momentum['vb'] = beta * self.momentum['vb'] + (1 - beta) * self.gradients['db']
        
        # 更新參數
        self.W -= learning_rate * self.momentum['vW']
        self.b -= learning_rate * self.momentum['vb']
    
    def get_parameters(self) -> Dict:
        """獲取層的參數"""
        return {
            'W': self.W.copy(),
            'b': self.b.copy()
        }
    
    def print_shape(self):
        """打印層的形狀信息"""
        print(f"\n層 {self.name} 的形狀信息:")
        print(f"W: {self.W.shape}")
        print(f"b: {self.b.shape}")
        if self.cache['A_prev'] is not None:
            print(f"A_prev: {self.cache['A_prev'].shape}")
        if self.cache['Z'] is not None:
            print(f"Z: {self.cache['Z'].shape}")
        if self.cache['A'] is not None:
            print(f"A: {self.cache['A'].shape}")

class ActivationLayer:
    def __init__(self, activation_type: str = "relu", alpha: float = 0.01):
        """
        初始化激活層
        Args:
            activation_type: 激活函數類型 ("relu", "sigmoid", "leaky_sigmoid")
            alpha: leaky sigmoid 的斜率參數
        """
        self.type = activation_type
        self.alpha = alpha
        self.cache = {'Z': None, 'A': None}
    
    def leaky_sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Leaky Sigmoid 激活函數
        Args:
            Z: 輸入值
        Returns:
            A: 激活值
        """
        sigmoid = 1 / (1 + np.exp(-Z))
        return np.where(sigmoid > 0.5, 
                       sigmoid, 
                       self.alpha * (sigmoid - 0.5) + 0.5)
    
    def leaky_sigmoid_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Leaky Sigmoid 的導數
        Args:
            Z: 輸入值
        Returns:
            derivative: 導數值
        """
        sigmoid = 1 / (1 + np.exp(-Z))
        sigmoid_derivative = sigmoid * (1 - sigmoid)
        return np.where(sigmoid > 0.5,
                       sigmoid_derivative,
                       self.alpha * sigmoid_derivative)
    
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        前向傳播
        Args:
            Z: 輸入值
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

class DetailedNeuralNetwork:
    def __init__(self, layer_dims: List[int]):
        """
        初始化神經網路
        Args:
            layer_dims: 各層的維度列表 [input_dim, hidden_dims..., output_dim]
        """
        self.layers = []
        
        # 創建層
        for l in range(1, len(layer_dims)):
            # 添加線性層
            layer = NeuralLayer(
                layer_dims[l-1], 
                layer_dims[l],
                name=f"Layer_{l}"
            )
            self.layers.append(layer)
            
            # 添加激活層
            activation = "sigmoid" if l == len(layer_dims)-1 else "relu"
            self.layers.append(ActivationLayer(activation))
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],      
            'val_accuracy': []    
        }

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        前向傳播
        Args:
            X: 輸入數據 (m x input_dim)
        Returns:
            A: 輸出預測 (m x output_dim)
        """
        A = X
        
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                Z = layer.linear_forward(A)
                A = Z  # 暫存
            else:  # ActivationLayer
                A = layer.forward(A)
        
        return A
    
    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray):
        """
        反向傳播
        Args:
            AL: 模型輸出
            Y: 真實標籤
        """
        m = Y.shape[0]
        
        # 計算輸出層的梯度
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        dA = dAL
        for layer in reversed(self.layers):
            if isinstance(layer, ActivationLayer):
                dZ = layer.backward(dA)
                dA = dZ  # 暫存
            else:  # NeuralLayer
                dA = layer.linear_backward(dA)
    
    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        計算損失和準確率
        Args:
            AL: 模型輸出
            Y: 真實標籤
        Returns:
            cost: 損失值
            accuracy: 準確率
        """
        m = Y.shape[0]
        epsilon = 1e-15
        
        # 計算交叉熵損失
        AL = np.clip(AL, epsilon, 1 - epsilon)
        cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        # 計算準確率
        predictions = (AL > 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        
        return cost, accuracy
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        評估模型在給定數據集上的表現
        """
        AL = self.forward_propagation(X)
        cost, accuracy = self.compute_cost(AL, Y)
        return cost, accuracy
    
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
                for layer in self.layers:
                    if isinstance(layer, NeuralLayer):
                        layer.update_parameters(learning_rate)
            
            # 計算平均訓練指標
            epoch_cost /= num_batches
            epoch_accuracy /= num_batches
            self.metrics['train_loss'].append(epoch_cost)
            self.metrics['train_accuracy'].append(epoch_accuracy)
            
            # 如果有驗證集，計算驗證指標
            if X_val is not None and Y_val is not None:
                val_cost, val_accuracy = self.evaluate(X_val, Y_val)
                self.metrics['val_loss'].append(val_cost)
                self.metrics['val_accuracy'].append(val_accuracy)
                
                if print_cost and epoch:
                    print(f"Epoch {epoch} | Train Loss: {epoch_cost:.4f} | Train Acc: {epoch_accuracy:.4f} | "
                          f"Val Loss: {val_cost:.4f} | Val Acc: {val_accuracy:.4f}")
            else:
                if print_cost and epoch:
                    print(f"Epoch {epoch} | Train Loss: {epoch_cost:.4f} | Train Acc: {epoch_accuracy:.4f}")
    
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        Args:
            X: 輸入數據
        Returns:
            predictions: 預測結果
        """
        AL = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
    
def create_difference_features(df):
    """創建差異特徵"""
    # 身體條件差異
    df['weight_diff'] = df['fighter1_weight'] - df['fighter2_weight']
    df['height_diff'] = df['fighter1_height'] - df['fighter2_height']
    df['reach_diff'] = df['fighter1_reach'] - df['fighter2_reach']
    df['age_diff'] = df['fighter1_age'] - df['fighter2_age']
    
    # 力量和經驗差異
    df['power_diff'] = df['fighter1_power_factor'] - df['fighter2_power_factor']
    df['exp_diff'] = df['fighter1_experience_factor'] - df['fighter2_experience_factor']
    return df

def create_ratio_features(df):
    """創建比率特徵"""
    # 勝率計算
    df['f1_total_fights'] = df['fighter1_wins'] + df['fighter1_losses']
    df['f2_total_fights'] = df['fighter2_wins'] + df['fighter2_losses']
    df['f1_win_rate'] = df['fighter1_wins'] / df['f1_total_fights']
    df['f2_win_rate'] = df['fighter2_wins'] / df['f2_total_fights']
    
    # KO 效率
    df['f1_ko_rate'] = df['fighter1_knockouts'] / df['fighter1_wins']
    df['f2_ko_rate'] = df['fighter2_knockouts'] / df['fighter2_wins']
    
    return df

def prepare_data():
    """準備訓練數據"""
    # 讀取數據
    df = pd.read_csv('data/matches.csv')
    
    # 特徵工程
    df = create_difference_features(df)
    df = create_ratio_features(df)
    
    # 選擇特徵
    features = [
        'weight_diff', 'height_diff', 'reach_diff', 'age_diff',
        'power_diff', 'exp_diff',
        'f1_win_rate', 'f2_win_rate',
        'f1_ko_rate', 'f2_ko_rate'
    ]
    
    # 準備特徵矩陣
    X = df[features].values
    
    # 準備標籤（假設fighter1是獲勝者的情況為1）
    Y = (df['winner'] == df['fighter1_name']).astype(int).values.reshape(-1, 1)
    
    # 分割數據
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, Y_train, Y_val, scaler, features

# 使用之前定義的DetailedNeuralNetwork類
if __name__ == "__main__":
    # 準備數據
    X_train, X_val, Y_train, Y_val, scaler, features = prepare_data()
    
    # 建立淺層神經網路（10個輸入特徵）
    layer_dims = [10, 16, 8, 1]  # 輸入層10個節點，兩個隱藏層，輸出層1個節點
    model = DetailedNeuralNetwork(layer_dims)
    
    # 訓練模型
    model.train(
        X_train, Y_train,
        X_val=X_val, 
        Y_val=Y_val,
        learning_rate=0.0001,  # 降低學習率以提高穩定性
        num_epochs=1000,
        batch_size=64,
        print_cost=True
    )
    
    # 繪製訓練指標
    model.plot_training_metrics()
    
    # 評估模型
    val_cost, val_accuracy = model.evaluate(X_val, Y_val)
    print(f"\n最終驗證集評估結果:")
    print(f"Loss: {val_cost:.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    
    # 預測示例
    def predict_fight(fighter1_data, fighter2_data, scaler, model):
        """預測對戰結果"""
        # 準備特徵
        fight_features = np.array([
            fighter1_data['weight'] - fighter2_data['weight'],
            fighter1_data['height'] - fighter2_data['height'],
            fighter1_data['reach'] - fighter2_data['reach'],
            fighter1_data['age'] - fighter2_data['age'],
            fighter1_data['power'] - fighter2_data['power'],
            fighter1_data['exp'] - fighter2_data['exp'],
            fighter1_data['wins'] / (fighter1_data['wins'] + fighter1_data['losses']),
            fighter2_data['wins'] / (fighter2_data['wins'] + fighter2_data['losses']),
            fighter1_data['knockouts'] / fighter1_data['wins'],
            fighter2_data['knockouts'] / fighter2_data['wins']
        ]).reshape(1, -1)
        
        # 標準化特徵
        fight_features_scaled = scaler.transform(fight_features)
        
        # 預測
        prediction = model.forward_propagation(fight_features_scaled)
        return prediction[0][0]
    
    # 測試預測
    fighter1 = {
        'weight': 75, 'height': 180, 'reach': 185, 'age': 28,
        'power': 1.1, 'exp': 0.8, 'wins': 15, 'losses': 2, 'knockouts': 10
    }
    
    fighter2 = {
        'weight': 73, 'height': 178, 'reach': 182, 'age': 25,
        'power': 0.9, 'exp': 0.7, 'wins': 12, 'losses': 3, 'knockouts': 8
    }
    
    win_prob = predict_fight(fighter1, fighter2, scaler, model)
    print(f"\n預測 Fighter1 獲勝機率: {win_prob:.3f}")