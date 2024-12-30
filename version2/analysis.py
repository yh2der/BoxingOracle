import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 顯示中文 (Windows)
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path: str = 'boxing-matches.csv') -> pd.DataFrame:
    """
    載入並回傳拳擊比賽資料。
    """
    df = pd.read_csv(csv_path)
    return df

def basic_analysis(df: pd.DataFrame):
    """
    進行基本統計與輸出，包含 KO率、判定率、回合數等
    """
    print("===== 基本統計 =====")
    total_matches = len(df)

    # 若資料中包含 TKO
    df['isKO'] = (df['victory_condition'] == 'KO').astype(int)
    df['isTKO'] = (df['victory_condition'] == 'TKO').astype(int)
    df['isDecision'] = df['victory_condition'].str.contains('Decision', case=False, na=False).astype(int)
    df['isDraw'] = (df['victory_condition'] == 'Draw').astype(int)

    ko_rate = df['isKO'].mean() * 100
    tko_rate = df['isTKO'].mean() * 100
    decision_rate = df['isDecision'].mean() * 100
    draw_rate = df['isDraw'].mean() * 100
    avg_rounds = df['rounds_completed'].mean()
    avg_hitA = df['hit_rate_A'].mean()
    avg_hitB = df['hit_rate_B'].mean()

    print(f"總場次: {total_matches}")
    print(f"KO率: {ko_rate:.2f}%")
    print(f"TKO率: {tko_rate:.2f}%")
    print(f"判定勝率: {decision_rate:.2f}%")
    print(f"平局率: {draw_rate:.2f}%")
    print(f"平均回合數: {avg_rounds:.2f}")
    print(f"平均命中率A: {avg_hitA:.2f}%")
    print(f"平均命中率B: {avg_hitB:.2f}%")

def plot_victory_distribution(df: pd.DataFrame):
    """
    勝利方式分佈 & 回合分佈
    """
    victory_counts = df['victory_condition'].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(victory_counts, labels=victory_counts.index, autopct='%1.1f%%')
    plt.title("勝利方式分佈")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.histplot(data=df, x='rounds_completed', bins=range(1, df['rounds_completed'].max()+2))
    plt.title("比賽回合數分佈")
    plt.show()

def derive_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    建立一些衍生欄位，例如:
    1. isAwin/isBwin
    2. power_diff, speed_diff, stamina_diff, defense_diff, chin_diff
    3. boxerA_BMI, boxerB_BMI (若 height 以 cm 為單位，需先轉 m)
    4. damage_per_punch_A, damage_per_punch_B
    5. isKO, isDecision, etc. (若尚未建立)
    """
    # isAwin, isBwin
    df['isAwin'] = df['winner'].str.contains('A').astype(int)
    df['isBwin'] = df['winner'].str.contains('B').astype(int)

    # 建立屬性差值
    df['power_diff'] = df['boxerA_power'] - df['boxerB_power']
    df['speed_diff'] = df['boxerA_speed'] - df['boxerB_speed']
    df['stamina_diff'] = df['boxerA_stamina'] - df['boxerB_stamina']
    df['defense_diff'] = df['boxerA_defense'] - df['boxerB_defense']
    df['chin_diff'] = df['boxerA_chin'] - df['boxerB_chin']
    df['experience_diff'] = df['boxerA_experience'] - df['boxerB_experience']

    # BMI (以 kg / (m^2)；若 height 單位是 cm，需 /100 才是 m)
    df['boxerA_BMI'] = df['boxerA_weight'] / ((df['boxerA_height']/100) ** 2)
    df['boxerB_BMI'] = df['boxerB_weight'] / ((df['boxerB_height']/100) ** 2)
    df['bmi_diff'] = df['boxerA_BMI'] - df['boxerB_BMI']

    # damage_per_punch
    # 要避免 total_punches 為 0
    df['damage_per_punch_A'] = df['total_damage_dealt_A'] / df['total_punches_A'].replace(0, np.nan)
    df['damage_per_punch_B'] = df['total_damage_dealt_B'] / df['total_punches_B'].replace(0, np.nan)

    # 若還沒定義 isKO / isDecision
    df['isKO'] = (df['victory_condition'] == 'KO').astype(int)
    df['isDecision'] = df['victory_condition'].str.contains('Decision', case=False, na=False).astype(int)

    return df

def analyze_attribute_vs_outcome(df: pd.DataFrame, attribute='boxerA_power'):
    """
    以分箱方式觀察 attribute 與 A 的勝率 / KO率 的關係
    預設以 boxerA_power 為例
    """
    # 分箱
    df['attr_bins'] = pd.cut(df[attribute], bins=5)
    grp = df.groupby('attr_bins')['isAwin'].mean()
    grp_ko = df.groupby('attr_bins')['isKO'].mean()

    plt.figure(figsize=(7,5))
    grp.plot(marker='o', label='A勝率', color='blue')
    grp_ko.plot(marker='s', label='KO率', color='red')
    plt.title(f"{attribute} 與 A的勝率/KO率(分箱)")
    plt.xlabel(f"{attribute} (分箱)")
    plt.ylabel("比率")
    plt.grid(True)
    plt.legend()
    plt.show()

def correlation_analysis(df: pd.DataFrame):
    """
    利用 Heatmap 觀察屬性 & 結果的線性相關
    """
    cols_of_interest = [
        'boxerA_power','boxerA_speed','boxerA_stamina','boxerA_defense','boxerA_chin','boxerA_experience',
        'boxerB_power','boxerB_speed','boxerB_stamina','boxerB_defense','boxerB_chin','boxerB_experience',
        'power_diff','speed_diff','stamina_diff','defense_diff','chin_diff','experience_diff',
        'bmi_diff','damage_per_punch_A','damage_per_punch_B',
        'final_hp_A','final_hp_B','final_stamina_A','final_stamina_B',
        'isAwin','isKO','isDecision','rounds_completed'
    ]

    # 取子集，排除缺失
    sub_df = df[cols_of_interest].dropna()
    corr_matrix = sub_df.corr()

    plt.figure(figsize=(14,12))
    sns.heatmap(corr_matrix, annot=False, cmap="YlGnBu")  # 若欄位多，annot=False 以免太擁擠
    plt.title("屬性 & 結果 相關係數 Heatmap")
    plt.show()

def logistic_regression_on_isAwin(df: pd.DataFrame):
    """
    Logistic Regression 分析：以 "isAwin" 為目標
    特徵可包含差值或其他衍生欄位
    """
    # 準備特徵與標籤
    features = [
        'power_diff','speed_diff','stamina_diff','defense_diff',
        'chin_diff','experience_diff','bmi_diff'
    ]
    X = df[features].copy()
    y = df['isAwin'].copy()

    # 去除空值
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # 加入截距
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    print("\n===== Logistic Regression (isAwin) =====")
    print(model.summary())

def random_forest_on_isAwin(df: pd.DataFrame):
    """
    隨機森林分析：以 "isAwin" 為目標，觀察特徵重要度
    """
    features = [
        'power_diff','speed_diff','stamina_diff','defense_diff',
        'chin_diff','experience_diff','bmi_diff','damage_per_punch_A','damage_per_punch_B'
    ]
    X = df[features].copy()
    y = df['isAwin'].copy()

    # 去除空值
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # 切分訓練/測試
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\n===== RandomForest on isAwin =====")
    print("混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))

    # 特徵重要度
    importances = rf.feature_importances_
    feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\n特徵重要度:")
    print(feature_importances)

    # 繪製特徵重要度
    plt.figure(figsize=(8,5))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("RandomForest 特徵重要度 (isAwin)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

def analyze_punch_patterns(df: pd.DataFrame):
    """
    分析出拳模式（Jab/Cross/Hook/Uppercut）與勝負、KO 等的關係
    示範分箱/比例
    """
    # 建立總出拳數
    punch_types = ['jab','cross','lead_hook','rear_hook','lead_uppercut','rear_uppercut']
    punch_colsA = [f"total_A_{pt}" for pt in punch_types]
    punch_colsB = [f"total_B_{pt}" for pt in punch_types]

    df['sum_punches_A'] = df[punch_colsA].sum(axis=1)
    df['sum_punches_B'] = df[punch_colsB].sum(axis=1)

    # 為了計算出拳比例，避免 0
    df['sum_punches_A'].replace(0, np.nan, inplace=True)
    df['sum_punches_B'].replace(0, np.nan, inplace=True)

    # jab_ratio 為例
    df['A_jab_ratio'] = df['total_A_jab'] / df['sum_punches_A']
    df['B_jab_ratio'] = df['total_B_jab'] / df['sum_punches_B']

    df['A_jab_bins'] = pd.cut(df['A_jab_ratio'], bins=5)
    jab_win_rates = df.groupby('A_jab_bins')['isAwin'].mean()
    jab_ko_rates = df.groupby('A_jab_bins')['isKO'].mean()

    plt.figure(figsize=(8,5))
    jab_win_rates.plot(marker='o', label='A勝率', color='blue')
    jab_ko_rates.plot(marker='s', label='KO率', color='red')
    plt.title("A的Jab使用比例 vs A勝率/KO率(分箱)")
    plt.xlabel("A_jab_ratio(分箱)")
    plt.ylabel("比率")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # 1) 讀取資料
    df = load_data('data/boxing-matches.csv')

    # 2) 基本統計
    basic_analysis(df)
    plot_victory_distribution(df)

    # 3) 建立衍生欄位
    df = derive_extra_features(df)

    # 4) 做一些屬性 vs 結果的視覺化
    #    這邊示範 boxerA_power，可換成 boxerB_power / power_diff / stamina_diff / ...
    analyze_attribute_vs_outcome(df, attribute='boxerA_power')

    # 5) 相關係數 Heatmap
    correlation_analysis(df)

    # 6) Logistic Regression (isAwin)
    logistic_regression_on_isAwin(df)

    # 7) Random Forest (isAwin)
    random_forest_on_isAwin(df)

    # 8) 出拳模式分析
    analyze_punch_patterns(df)

    print("\n分析流程全部完成！")

if __name__ == "__main__":
    main()
