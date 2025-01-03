##################################################
# 1. Imports
##################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

##################################################
# 2. Data Preprocessing & Feature Engineering
##################################################
def prepare_static_data(df):
    """
    對原始數據進行數據預處理，加入更多拳擊相關的特徵
    
    Args:
        df: 原始數據DataFrame
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    # 1. 計算基本特徵差異
    static_attributes = ['weight', 'height', 'reach', 'power', 'speed', 
                        'stamina', 'defense', 'chin', 'experience']
    
    for attr in static_attributes:
        df[f'{attr}_diff'] = df[f'boxerA_{attr}'] - df[f'boxerB_{attr}']
        
        # 計算比率
        if attr in ['power', 'speed', 'stamina', 'defense', 'chin']:
            df[f'{attr}_ratio'] = df[f'boxerA_{attr}'] / df[f'boxerB_{attr}']
    
    # 2. 綜合實力分數 (調整權重示範)
    def calculate_fighter_score(row, prefix):
        return (
            row[f'{prefix}_power'] * 0.30 +
            row[f'{prefix}_speed'] * 0.20 +
            row[f'{prefix}_stamina'] * 0.15 +
            row[f'{prefix}_defense'] * 0.15 +
            row[f'{prefix}_chin'] * 0.10 +
            row[f'{prefix}_experience'] * 0.10
        )
    
    # 3. 物理優勢指標
    def calculate_physical_advantage(row):
        weight_adv = (row['boxerA_weight'] - row['boxerB_weight']) * 0.005
        reach_height_ratio_A = row['boxerA_reach'] / row['boxerA_height']
        reach_height_ratio_B = row['boxerB_reach'] / row['boxerB_height']
        reach_advantage = (reach_height_ratio_A - reach_height_ratio_B) * 5
        bmi_A = row['boxerA_weight'] / ((row['boxerA_height']/100) ** 2)
        bmi_B = row['boxerB_weight'] / ((row['boxerB_height']/100) ** 2)
        bmi_advantage = (bmi_A - bmi_B) * 0.1
        return weight_adv + reach_advantage + bmi_advantage
    
    # 4. 技術優勢指標
    def calculate_technical_advantage(row):
        speed_power_ratio_A = row['boxerA_speed'] / row['boxerA_power']
        speed_power_ratio_B = row['boxerB_speed'] / row['boxerB_power']
        speed_power_adv = (speed_power_ratio_A - speed_power_ratio_B) * 10
        
        defense_efficiency_A = row['boxerA_defense'] * row['boxerA_chin'] / 100
        defense_efficiency_B = row['boxerB_defense'] * row['boxerB_chin'] / 100
        defense_adv = (defense_efficiency_A - defense_efficiency_B) * 0.02
        
        experience_modifier = (row['boxerA_experience'] - row['boxerB_experience']) * 0.01
        return speed_power_adv + defense_adv + experience_modifier
    
    df['fighter_score_A'] = df.apply(lambda x: calculate_fighter_score(x, 'boxerA'), axis=1)
    df['fighter_score_B'] = df.apply(lambda x: calculate_fighter_score(x, 'boxerB'), axis=1)
    df['fighter_score_diff'] = df['fighter_score_A'] - df['fighter_score_B']
    df['physical_advantage'] = df.apply(calculate_physical_advantage, axis=1)
    df['technical_advantage'] = df.apply(calculate_technical_advantage, axis=1)
    
    # 5. 建立 feature_columns
    feature_columns = [
        'weight_diff', 'height_diff', 'reach_diff',
        'power_diff', 'speed_diff', 'stamina_diff', 
        'defense_diff', 'chin_diff', 'experience_diff',
        'power_ratio', 'speed_ratio', 'stamina_ratio', 
        'defense_ratio', 'chin_ratio',
        'fighter_score_diff',
        'physical_advantage',
        'technical_advantage'
    ]
    
    # 6. 轉換標籤
    df['winner_encoded'] = (df['winner'].str.contains('A')).astype(int)
    
    # 7. 分割數據
    X = df[feature_columns]
    y = df['winner_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 8. 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def calculate_physical_advantage(row):
    """計算選手之間的體格優勢"""
    weight_adv = (row['boxerA_weight'] - row['boxerB_weight']) * 0.005
    
    reach_height_ratio_A = row['boxerA_reach'] / row['boxerA_height']
    reach_height_ratio_B = row['boxerB_reach'] / row['boxerB_height']
    reach_advantage = (reach_height_ratio_A - reach_height_ratio_B) * 5
    
    bmi_A = row['boxerA_weight'] / ((row['boxerA_height']/100) ** 2)
    bmi_B = row['boxerB_weight'] / ((row['boxerB_height']/100) ** 2)
    bmi_advantage = (bmi_A - bmi_B) * 0.1
    
    return weight_adv + reach_advantage + bmi_advantage

def calculate_technical_advantage(row):
    """計算選手之間的技術優勢"""
    speed_power_ratio_A = row['boxerA_speed'] / row['boxerA_power']
    speed_power_ratio_B = row['boxerB_speed'] / row['boxerB_power']
    speed_power_adv = (speed_power_ratio_A - speed_power_ratio_B) * 10
    
    defense_efficiency_A = row['boxerA_defense'] * row['boxerA_chin'] / 100
    defense_efficiency_B = row['boxerB_defense'] * row['boxerB_chin'] / 100
    defense_adv = (defense_efficiency_A - defense_efficiency_B) * 0.02
    
    experience_modifier = (row['boxerA_experience'] - row['boxerB_experience']) * 0.01
    
    return speed_power_adv + defense_adv + experience_modifier

def calculate_fighter_score(row, prefix):
    """計算選手的綜合實力分數"""
    return (
        row[f'{prefix}_power'] * 0.25 +
        row[f'{prefix}_speed'] * 0.20 +
        row[f'{prefix}_stamina'] * 0.15 +
        row[f'{prefix}_defense'] * 0.20 +
        row[f'{prefix}_chin'] * 0.10 +
        row[f'{prefix}_experience'] * 0.10
    )

##################################################
# 3. Neural Network Classes
##################################################
class NeuralLayer:
    def __init__(self, input_dim: int, output_dim: int, name: str = ""):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Xavier初始化
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))
        
        self.gradients = {'dW': np.zeros_like(self.W), 'db': np.zeros_like(self.b)}
        
        # Adam 參數
        self.adam = {
            'mW': np.zeros_like(self.W),
            'vW': np.zeros_like(self.W),
            'mb': np.zeros_like(self.b),
            'vb': np.zeros_like(self.b)
        }
        
    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        return self.Z
    
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        m = self.A_prev.shape[0]
        
        self.gradients['dW'] = (1/m) * np.dot(self.A_prev.T, dZ)
        self.gradients['db'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # 梯度裁剪 (可視需求保留或刪除)
        clip_value = 5.0
        self.gradients['dW'] = np.clip(self.gradients['dW'], -clip_value, clip_value)
        self.gradients['db'] = np.clip(self.gradients['db'], -clip_value, clip_value)
        
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev
    
    def update_parameters(self, learning_rate: float, iteration: int,
                         beta1: float = 0.9, beta2: float = 0.999,
                         epsilon: float = 1e-8):
        # Adam
        self.adam['mW'] = beta1 * self.adam['mW'] + (1 - beta1) * self.gradients['dW']
        self.adam['vW'] = beta2 * self.adam['vW'] + (1 - beta2) * np.square(self.gradients['dW'])
        self.adam['mb'] = beta1 * self.adam['mb'] + (1 - beta1) * self.gradients['db']
        self.adam['vb'] = beta2 * self.adam['vb'] + (1 - beta2) * np.square(self.gradients['db'])
        
        mW_corrected = self.adam['mW'] / (1 - beta1 ** iteration)
        vW_corrected = self.adam['vW'] / (1 - beta2 ** iteration)
        mb_corrected = self.adam['mb'] / (1 - beta1 ** iteration)
        vb_corrected = self.adam['vb'] / (1 - beta2 ** iteration)
        
        self.W -= learning_rate * mW_corrected / (np.sqrt(vW_corrected) + epsilon)
        self.b -= learning_rate * mb_corrected / (np.sqrt(vb_corrected) + epsilon)

class ActivationLayer:
    def __init__(self, activation_type: str = "relu", alpha: float = 0.01):
        self.type = activation_type
        self.alpha = alpha
        self.cache = {'Z': None, 'A': None}
    
    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache['Z'] = Z
        
        if self.type == "relu":
            A = np.maximum(0, Z)
        elif self.type == "leaky_sigmoid":
            A = self.leaky_sigmoid(Z)
        else:  # default: sigmoid
            A = 1 / (1 + np.exp(-Z))
        
        self.cache['A'] = A
        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.type == "relu":
            dZ = dA * (self.cache['Z'] > 0)
        elif self.type == "leaky_sigmoid":
            dZ = dA * self.leaky_sigmoid_derivative(self.cache['Z'])
        else:  # default: sigmoid
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        return dZ
    
    def leaky_sigmoid(self, Z):
        # 只是一個自定義示範
        return 1 / (1 + np.exp(-Z)) + self.alpha * Z
    
    def leaky_sigmoid_derivative(self, Z):
        # 自定義示範
        sig = 1 / (1 + np.exp(-Z))
        return sig * (1 - sig) + self.alpha

class DetailedNeuralNetwork:    
    def __init__(self, layer_dims: List[int], lambda_l2: float = 0.01):
        self.layer_dims = layer_dims  # Add this for model reconstruction
        self.layers = []
        self.lambda_l2 = lambda_l2
        for l in range(1, len(layer_dims)):
            layer = NeuralLayer(layer_dims[l-1], layer_dims[l], f"Layer_{l}")
            self.layers.append(layer)
            
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
    
    def save_model(self, filepath: str):
        """
        儲存模型參數到檔案
        
        Args:
            filepath: 儲存路徑，建議使用 .pth 副檔名
        """
        model_state = {
            'layer_dims': self.layer_dims,
            'lambda_l2': self.lambda_l2,
            'layers': []
        }
        
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                layer_state = {
                    'type': 'neural',
                    'W': layer.W,
                    'b': layer.b,
                    'input_dim': layer.input_dim,
                    'output_dim': layer.output_dim,
                    'name': layer.name
                }
            else:  # ActivationLayer
                layer_state = {
                    'type': 'activation',
                    'activation_type': layer.type,
                    'alpha': layer.alpha
                }
            model_state['layers'].append(layer_state)
        
        np.save(filepath, model_state, allow_pickle=True)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        從檔案載入模型參數
        
        Args:
            filepath: 模型檔案路徑
        Returns:
            DetailedNeuralNetwork: 載入參數後的模型
        """
        model_state = np.load(filepath, allow_pickle=True).item()
        
        # 重建模型架構
        model = cls(model_state['layer_dims'], model_state['lambda_l2'])
        model.layers = []  # 清空預設初始化的層
        
        # 載入每一層的參數
        for layer_state in model_state['layers']:
            if layer_state['type'] == 'neural':
                layer = NeuralLayer(
                    layer_state['input_dim'],
                    layer_state['output_dim'],
                    layer_state['name']
                )
                layer.W = layer_state['W']
                layer.b = layer_state['b']
                model.layers.append(layer)
            else:  # activation
                layer = ActivationLayer(
                    layer_state['activation_type'],
                    layer_state['alpha']
                )
                model.layers.append(layer)
        
        print(f"Model loaded from {filepath}")
        return model

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray, 
                    compute_regularization: bool = True) -> Tuple[float, float]:
        m = Y.shape[0]
        epsilon = 1e-15
        
        AL = np.clip(AL, epsilon, 1 - epsilon)
        cross_entropy_cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        l2_cost = 0
        if compute_regularization:
            for layer in self.layers:
                if isinstance(layer, NeuralLayer):
                    l2_cost += np.sum(np.square(layer.W))
            l2_cost = (self.lambda_l2 / (2 * m)) * l2_cost
        
        cost = cross_entropy_cost + l2_cost
        
        predictions = (AL > 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        
        return cost, accuracy
    
    def forward_propagation(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A, training)
        return A
    
    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray):
        m = Y.shape[0]
        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
        
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL
        
        for layer in reversed(self.layers):
            if isinstance(layer, ActivationLayer):
                dZ = layer.backward(dA)
                dA = dZ
            else:  # NeuralLayer
                dA = layer.backward(dA)
                layer.gradients['dW'] += (self.lambda_l2 / m) * layer.W
    
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
             X_val: np.ndarray = None, Y_val: np.ndarray = None,
             learning_rate: float = 0.01,
             num_epochs: int = 100,
             batch_size: int = 32,
             print_cost: bool = True):
        
        m = X_train.shape[0]
        iteration = 0
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            epoch_accuracy = 0
            num_batches = int(np.ceil(m / batch_size))
            
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]
            
            for batch in range(num_batches):
                iteration += 1
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                AL = self.forward_propagation(X_batch)
                batch_cost, batch_accuracy = self.compute_cost(AL, Y_batch)
                epoch_cost += batch_cost
                epoch_accuracy += batch_accuracy
                
                self.backward_propagation(AL, Y_batch)
                
                current_lr = learning_rate / (1 + 0.01 * epoch)  # 學習率遞減
                for layer in self.layers:
                    if isinstance(layer, NeuralLayer):
                        layer.update_parameters(current_lr, iteration)
            
            epoch_cost /= num_batches
            epoch_accuracy /= num_batches
            self.metrics['train_loss'].append(epoch_cost)
            self.metrics['train_accuracy'].append(epoch_accuracy)
            
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
        AL = self.forward_propagation(X, training=False)
        predictions = (AL > 0.5).astype(int)
        return predictions
    
    def plot_training_metrics(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        if len(self.metrics['val_loss']) > 0:
            plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
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

##################################################
# 4. Basic Analysis & Advanced Visualization
##################################################
def analyze_training_data(X_train, y_train, feature_columns):
    """
    分析訓練數據中的特徵關係。
    1) 對特徵做分組後的分佈圖
    2) 計算相關性, effect size, win_rate_impact
    """
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df['winner'] = y_train
    
    feature_groups = {
        'Physical Attributes': ['weight_diff', 'height_diff', 'reach_diff'],
        'Combat Stats': ['power_diff', 'speed_diff', 'stamina_diff', 'defense_diff', 'chin_diff'],
        'Technical Ratios': ['power_ratio', 'speed_ratio', 'stamina_ratio', 'defense_ratio', 'chin_ratio'],
        'Composite Scores': ['fighter_score_diff', 'physical_advantage', 'technical_advantage']
    }
    
    # 1. 分組繪圖 (histogram)
    for group_name, features in feature_groups.items():
        plt.figure(figsize=(15, 5 * ((len(features) + 2) // 3)))
        for i, feature in enumerate(features, 1):
            if feature in train_df.columns:
                plt.subplot((len(features) + 2) // 3, 3, i)
                sns.histplot(data=train_df, x=feature, hue='winner', bins=30, alpha=0.6)
                plt.title(f'{feature} Distribution', fontsize=10)
        plt.suptitle(f'{group_name} Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 2. 計算特徵的重要度（相關性, Cohen’s d, win_rate_impact）
    feature_importance = []
    for feature in feature_columns:
        correlation = train_df[feature].corr(train_df['winner'])
        
        group1 = train_df[train_df['winner'] == 1][feature]
        group0 = train_df[train_df['winner'] == 0][feature]
        
        # Cohen's d
        cohens_d = (group1.mean() - group0.mean()) / np.sqrt(
            ((group1.std() ** 2 + group0.std() ** 2) / 2)
        ) if (group1.std() != 0 and group0.std() != 0) else 0
        
        # 勝率影響
        median_val = train_df[feature].median()
        high_values = train_df[train_df[feature] > median_val]['winner'].mean()
        low_values = train_df[train_df[feature] <= median_val]['winner'].mean()
        win_rate_impact = abs(high_values - low_values)
        
        feature_importance.append({
            'feature': feature,
            'correlation': abs(correlation),
            'effect_size': abs(cohens_d),
            'win_rate_impact': win_rate_impact
        })
    
    fi_df = pd.DataFrame(feature_importance).sort_values('effect_size', ascending=False)
    return fi_df

def plot_correlation_heatmap(df, feature_columns):
    """
    繪製特徵與 winner 的相關性熱力圖
    """
    plt.figure(figsize=(12, 10))
    corr_cols = feature_columns + ['winner']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap of Features and Winner")
    plt.show()

def plot_pairplot(df, features):
    """
    對指定的 features 做 PairPlot，觀察其與 winner 的關係
    """
    sns.pairplot(df, vars=features, hue='winner', diag_kind='kde', corner=True)
    plt.suptitle("PairPlot of Key Features", y=1.02)
    plt.show()

def plot_box_violin_by_winner(df, features, plot_type="box"):
    """
    繪製 Box Plot 或 Violin Plot，比較 winner=0/1 的特徵分佈
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, feature in enumerate(features, start=1):
        plt.subplot(n_rows, n_cols, i)
        if plot_type == "box":
            sns.boxplot(data=df, x='winner', y=feature)
        else:
            sns.violinplot(data=df, x='winner', y=feature, inner="box")
        plt.title(f"{feature} by Winner")
    plt.tight_layout()
    plt.show()

def plot_win_rate_by_bins(df, feature, bins=5):
    """
    將 feature 分桶後，觀察各桶的勝率 (winner=1 的平均)
    """
    temp_df = df.copy()
    temp_df['bin'] = pd.cut(temp_df[feature], bins=bins, include_lowest=True)
    grouped = temp_df.groupby('bin')['winner'].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x='bin', y='winner', data=grouped)
    plt.ylim(0, 1)
    plt.title(f"Win Rate by {feature} Bins")
    plt.ylabel("Win Rate")
    plt.xlabel(feature)
    plt.show()

def plot_2d_bin_winrate(df, feature_x, feature_y, bins_x=5, bins_y=5):
    """
    同時對 feature_x, feature_y 各分桶，計算該網格內的勝率並以 Heatmap 呈現
    """
    temp_df = df.copy()
    temp_df['bin_x'] = pd.cut(temp_df[feature_x], bins=bins_x)
    temp_df['bin_y'] = pd.cut(temp_df[feature_y], bins=bins_y)
    
    pivot_df = temp_df.pivot_table(
        index='bin_y', columns='bin_x', values='winner', aggfunc='mean'
    )
    
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Win Rate by {feature_x} & {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

##################################################
# 5. Model Analyzer
##################################################
class ModelAnalyzer:
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
        self.prediction_analysis = None

    def analyze_predictions(self, X_scaled, y_true):
        y_prob = self.model.forward_propagation(X_scaled, training=False)
        y_pred = (y_prob > 0.5).astype(int).squeeze()
        y_prob = y_prob.squeeze()
        
        if len(y_true.shape) > 1:
            y_true = y_true.squeeze()

        pred_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        pred_df['true_label'] = y_true
        pred_df['predicted_prob'] = y_prob
        pred_df['predicted_label'] = y_pred
        pred_df['correct_prediction'] = (y_pred == y_true).astype(int)

        self.prediction_analysis = pred_df

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        # 預測機率的散佈圖
        plt.figure(figsize=(15, 5*((len(self.feature_columns)+2)//3)))
        for i, feature in enumerate(self.feature_columns, 1):
            plt.subplot((len(self.feature_columns)+2)//3, 3, i)
            sns.scatterplot(data=pred_df, x=feature, y='predicted_prob', 
                            hue='true_label', alpha=0.5)
            plt.title(f'{feature} vs Prediction Probability')
        plt.tight_layout()
        plt.show()
        
        return pred_df

    def analyze_errors(self):
        if self.prediction_analysis is None:
            raise ValueError("必須先執行 analyze_predictions")
        
        errors = self.prediction_analysis[self.prediction_analysis['correct_prediction'] == 0]
        correct = self.prediction_analysis[self.prediction_analysis['correct_prediction'] == 1]

        plt.figure(figsize=(15, 5))
        
        # 錯誤預測的機率分布
        plt.subplot(1, 3, 1)
        sns.histplot(data=errors, x='predicted_prob', bins=20)
        plt.title('Error Cases Probability Distribution')

        # 正確預測的機率分布
        plt.subplot(1, 3, 2)
        sns.histplot(data=correct, x='predicted_prob', bins=20)
        plt.title('Correct Cases Probability Distribution')

        # 錯誤率隨預測機率變化
        plt.subplot(1, 3, 3)
        prob_bins = pd.cut(self.prediction_analysis['predicted_prob'], bins=10)
        error_rates = 1 - self.prediction_analysis.groupby(prob_bins)['correct_prediction'].mean()
        # prob_bins 只是分段標籤，畫圖時可自己定義 x 座標
        bin_centers = np.linspace(0, 1, 10)
        plt.plot(bin_centers, error_rates, marker='o')
        plt.title('Error Rate vs Prediction Probability')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Error Rate')
        
        plt.tight_layout()
        plt.show()

        return errors

##################################################
# 6. Main Execution Flow
##################################################
if __name__ == "__main__":
    model_path = 'model/boxing_predictor.pth'
    
    if os.path.exists(model_path):
        print("載入已訓練的模型...")
        model = DetailedNeuralNetwork.load_model(model_path)
        df_for_preprocessing = pd.read_csv('data/boxing-matches.csv')  # 與原訓練資料結構相同
        _, _, _, _, scaler, feature_columns = prepare_static_data(df_for_preprocessing)
    else:
        print("開始訓練新模型...")
        # 1. 讀取資料
        df = pd.read_csv('data/boxing-matches.csv')
        print("原始數據形狀:", df.shape)

        # 2. 準備數據
        X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_static_data(df)
        
        # 3. 建立並訓練模型
        model = DetailedNeuralNetwork(
            layer_dims=[X_train.shape[1], 32, 16, 1],
            lambda_l2=0.00001
        )
        model.train(
            X_train, np.array(y_train).reshape(-1, 1),
            X_val=X_test, Y_val=np.array(y_test).reshape(-1, 1),
            learning_rate=0.0001,
            num_epochs=200,
            batch_size=32
        )
        
        # 4. 儲存模型
        os.makedirs('model', exist_ok=True)
        model.save_model(model_path)
    
    # ================== 單場比賽測試 ==================
    tyson = {
        'name': "Iron Mike Tyson",
        'weight': 95.5,
        'height': 178,
        'reach': 180,
        'power': 95,
        'speed': 90,
        'stamina': 85,
        'defense': 80,
        'chin': 85,
        'experience': 95
    }

    ali = {
        'name': "Muhammad Ali",
        'weight': 92.0,
        'height': 191,
        'reach': 198,
        'power': 88,
        'speed': 95,
        'stamina': 95,
        'defense': 90,
        'chin': 90,
        'experience': 98
    }

    # 準備測試資料...
    test_match = pd.DataFrame({
        # 拳擊手A：Mike Tyson
        'boxerA_weight': [tyson['weight']],
        'boxerA_height': [tyson['height']],
        'boxerA_reach': [tyson['reach']],
        'boxerA_power': [tyson['power']],
        'boxerA_speed': [tyson['speed']],
        'boxerA_stamina': [tyson['stamina']],
        'boxerA_defense': [tyson['defense']],
        'boxerA_chin': [tyson['chin']],
        'boxerA_experience': [tyson['experience']],

        # 拳擊手B：Muhammad Ali
        'boxerB_weight': [ali['weight']],
        'boxerB_height': [ali['height']],
        'boxerB_reach': [ali['reach']],
        'boxerB_power': [ali['power']],
        'boxerB_speed': [ali['speed']],
        'boxerB_stamina': [ali['stamina']],
        'boxerB_defense': [ali['defense']],
        'boxerB_chin': [ali['chin']],
        'boxerB_experience': [ali['experience']]
    })

    # 計算所有必要的特徵...
    for attr in ['weight', 'height', 'reach', 'power', 'speed', 
                'stamina', 'defense', 'chin', 'experience']:
        test_match[f'{attr}_diff'] = test_match[f'boxerA_{attr}'] - test_match[f'boxerB_{attr}']
        
        if attr in ['power', 'speed', 'stamina', 'defense', 'chin']:
            test_match[f'{attr}_ratio'] = (
                test_match[f'boxerA_{attr}'] / test_match[f'boxerB_{attr}']
            )

    # 計算綜合指標...
    test_match['physical_advantage'] = test_match.apply(calculate_physical_advantage, axis=1)
    test_match['technical_advantage'] = test_match.apply(calculate_technical_advantage, axis=1)
    
    def calculate_fighter_score_local(row, prefix):
        return (
            row[f'{prefix}_power'] * 0.25 +
            row[f'{prefix}_speed'] * 0.20 +
            row[f'{prefix}_stamina'] * 0.15 +
            row[f'{prefix}_defense'] * 0.20 +
            row[f'{prefix}_chin'] * 0.10 +
            row[f'{prefix}_experience'] * 0.10
        )

    test_match['fighter_score_A'] = test_match.apply(lambda x: calculate_fighter_score_local(x, 'boxerA'), axis=1)
    test_match['fighter_score_B'] = test_match.apply(lambda x: calculate_fighter_score_local(x, 'boxerB'), axis=1)
    test_match['fighter_score_diff'] = test_match['fighter_score_A'] - test_match['fighter_score_B']

    # 準備預測特徵並進行預測
    X_demo = test_match[feature_columns]
    X_demo_scaled = scaler.transform(X_demo)  # 注意：在實際使用時需要保存scaler

    demo_pred = model.predict(X_demo_scaled)
    demo_prob = model.forward_propagation(X_demo_scaled, training=False)

    # 輸出預測結果
    print("\n====== 拳擊手 Tyson vs 拳擊手 Ali 模擬對決 ======")
    print(f"獲勝預測: {'Tyson' if demo_pred[0][0] == 1 else 'Ali'}")
    print(f"拳擊手 Tyson 獲勝機率: {demo_prob[0][0]*100:.2f}%")
    print(f"拳擊手 Ali 獲勝機率: {(1-demo_prob[0][0])*100:.2f}%")

    print("\n====== 關鍵屬性差異 ======")
    for attr in ['weight', 'height', 'reach', 'power', 'speed', 
                 'stamina', 'defense', 'chin', 'experience']:
        diff = test_match[f'{attr}_diff'].values[0]
        print(f"{attr}: {'+' if diff > 0 else ''}{diff:.1f}")

    print("\n====== 綜合實力評分 ======")
    print(f"拳擊手 Tyson: {test_match['fighter_score_A'].values[0]:.2f}")
    print(f"拳擊手 Ali: {test_match['fighter_score_B'].values[0]:.2f}")
