# BoxingOracle v1

基礎版拳擊比賽模擬系統，實現了基本的拳擊規則計算與數據收集功能。

## 核心功能

### 1. 規則計算
- 真實的拳擊傷害計算
- 體力消耗系統
- 出拳速度模擬
- 命中率計算
- 得分系統

### 2. 選手屬性系統
```python
@dataclass
class FighterStats:
    name: str
    weight: float  # kg
    height: float  # cm
    reach: float  # cm
    age: int
    wins: int
    losses: int
    knockouts: int
```

### 3. 戰鬥機制
- 三種基本拳種：
  - 直拳 (基礎傷害: 12-22)
  - 勾拳 (基礎傷害: 20-30)
  - 重拳 (基礎傷害: 30-45)
- 防禦系統：
  - 格擋判定
  - 反擊機制
- 體力管理：
  - 回合休息恢復
  - 戰鬥消耗

### 4. 數據收集
- 即時戰鬥記錄
- CSV 格式儲存
- 詳細的比賽數據

## 系統架構

### 主要類別
1. `FighterStats`: 選手基礎屬性
2. `Fighter`: 戰鬥單位實現
3. `BoxingRound`: 回合控制器
4. `OneRoundBoxingGame`: 比賽管理器
5. `MatchLogger`: 數據記錄器

## 使用方法

### 1. 建立選手
```python
fighter_stats = FighterStats(
    name="張三",
    weight=75,
    height=175, 
    reach=180,
    age=28,
    wins=5,
    losses=2,
    knockouts=3
)
```

### 2. 執行單場比賽
```python
game = OneRoundBoxingGame(fighter1_stats, fighter2_stats)
await game.start_game()
```

### 3. 大量比賽模擬
```python
async def main():
    fighters_stats_list = [
        FighterStats(...),
        FighterStats(...),
        # ...
    ]
    
    logger = MatchLogger()
    num_matches = 1000
    
    for i in range(num_matches):
        fighter1, fighter2 = random.sample(fighters_stats_list, 2)
        game = OneRoundBoxingGame(fighter1, fighter2, i+1, logger)
        await game.start_game()
```

## 數據格式

### 1. Event Log (boxing_events_log.csv)
- match_id
- action_id
- timestamp
- attacker_name
- defender_name
- action
- damage
- hp_values
- stamina_values
- combo_status

### 2. Match Summary (boxing_match_summary.csv)
- match_id
- match_date
- winner
- fighter_stats
- final_status
- ko_happened

## 限制與待改進
1. 單回合制限制
2. 簡化的物理引擎
3. 基礎的數據收集
4. 缺乏預測功能