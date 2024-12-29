import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df):
    """
    對原始數據進行所有的數據預處理和特徵工程
    
    Args:
        df: 原始數據DataFrame
    Returns:
        X_train, X_test, y_train, y_test, scaler: 處理後的訓練和測試數據
    """
    # Step 1: 計算基本特徵差異
    attributes = ['weight', 'height', 'reach', 'power', 'speed', 
                 'stamina', 'defense', 'chin', 'experience']
    
    for attr in attributes:
        df[f'{attr}_diff'] = df[f'boxerA_{attr}'] - df[f'boxerB_{attr}']
    
    # Step 2: 計算每一擊的傷害和效率
    eps = 1e-7  # 避免除以零
    
    # 每一擊的傷害
    df['damage_per_hit_A'] = df['total_damage_dealt_A'] / (df['total_hits_A'] + eps)
    df['damage_per_hit_B'] = df['total_damage_dealt_B'] / (df['total_hits_B'] + eps)
    df['damage_per_hit_diff'] = df['damage_per_hit_A'] - df['damage_per_hit_B']
    
    # 體力使用效率
    df['damage_per_stamina_A'] = df['total_damage_dealt_A'] / (df['total_stamina_used_A'] + eps)
    df['damage_per_stamina_B'] = df['total_damage_dealt_B'] / (df['total_stamina_used_B'] + eps)
    df['stamina_efficiency_diff'] = df['damage_per_stamina_A'] - df['damage_per_stamina_B']
    
    # 命中率差異
    df['hit_rate_diff'] = df['hit_rate_A'] - df['hit_rate_B']
    
    # Step 3: 計算每種拳擊的使用比例
    for prefix in ['A', 'B']:
        total_punches = df[f'total_punches_{prefix}'] + eps
        
        # 基本拳擊比例
        df[f'jab_ratio_{prefix}'] = df[f'total_{prefix}_jab'] / total_punches
        df[f'cross_ratio_{prefix}'] = df[f'total_{prefix}_cross'] / total_punches
        
        # 組合拳擊比例
        df[f'hook_ratio_{prefix}'] = (df[f'total_{prefix}_lead_hook'] + 
                                    df[f'total_{prefix}_rear_hook']) / total_punches
        df[f'uppercut_ratio_{prefix}'] = (df[f'total_{prefix}_lead_uppercut'] + 
                                        df[f'total_{prefix}_rear_uppercut']) / total_punches
    
    # Step 4: 計算綜合實力分數
    def calculate_fighter_score(row, prefix):
        return (
            row[f'{prefix}_power'] * 0.25 +
            row[f'{prefix}_speed'] * 0.20 +
            row[f'{prefix}_stamina'] * 0.15 +
            row[f'{prefix}_defense'] * 0.20 +
            row[f'{prefix}_chin'] * 0.10 +
            row[f'{prefix}_experience'] * 0.10
        )
    
    df['fighter_score_A'] = df.apply(lambda x: calculate_fighter_score(x, 'boxerA'), axis=1)
    df['fighter_score_B'] = df.apply(lambda x: calculate_fighter_score(x, 'boxerB'), axis=1)
    df['fighter_score_diff'] = df['fighter_score_A'] - df['fighter_score_B']
    
    # Step 5: 計算實戰表現分數
    def calculate_performance_score(row, prefix):
        return (
            row[f'hit_rate_{prefix}'] * 0.3 +
            row[f'final_hp_{prefix}'] / 100 * 0.3 +
            row[f'final_stamina_{prefix}'] / 100 * 0.2 +
            (row[f'total_knockdowns_{prefix}'] > 0) * 0.2
        )
    
    df['performance_score_A'] = df.apply(lambda x: calculate_performance_score(x, 'A'), axis=1)
    df['performance_score_B'] = df.apply(lambda x: calculate_performance_score(x, 'B'), axis=1)
    df['performance_score_diff'] = df['performance_score_A'] - df['performance_score_B']
    
    # Step 6: 準備特徵矩陣
    feature_columns = [
        # 基本屬性差異
        'weight_diff', 'height_diff', 'reach_diff',
        'power_diff', 'speed_diff', 'stamina_diff', 
        'defense_diff', 'chin_diff', 'experience_diff',
        
        # 效率差異
        'hit_rate_diff', 'damage_per_hit_diff', 'stamina_efficiency_diff',
        
        # 戰鬥風格差異
        'jab_ratio_A', 'jab_ratio_B',
        'cross_ratio_A', 'cross_ratio_B',
        'hook_ratio_A', 'hook_ratio_B',
        'uppercut_ratio_A', 'uppercut_ratio_B',
        
        # 綜合分數差異
        'fighter_score_diff', 'performance_score_diff'
    ]
    
    # Step 7: 準備標籤
    df['winner_encoded'] = (df['winner'].str.contains('A')).astype(int)
    
    # Step 8: 分割數據並標準化
    X = df[feature_columns]
    y = df['winner_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main():
    """
    測試數據準備流程
    """
    # 讀取數據
    df = pd.read_csv('boxing-matches.csv')
    print("原始數據形狀:", df.shape)
    
    # 準備數據
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    print("\n處理後的數據形狀:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

if __name__ == "__main__":
    main()