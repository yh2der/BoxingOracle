import random
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from pathlib import Path

class PunchType(Enum):
    JAB = "Jab"
    CROSS = "Cross" 
    LEAD_HOOK = "Lead Hook"
    REAR_HOOK = "Rear Hook"
    LEAD_UPPERCUT = "Lead Uppercut"
    REAR_UPPERCUT = "Rear Uppercut"

PUNCH_PROPERTIES = {
    PunchType.JAB: {
        "damage": 25,
        "stamina_cost": 0.3,
        "time_cost": 0.3,
        "accuracy": 0.9,
        "points": 1
    },
    PunchType.CROSS: {
        "damage": 40,
        "stamina_cost": 0.6,
        "time_cost": 0.5,
        "accuracy": 0.8,
        "points": 2
    },
    PunchType.LEAD_HOOK: {
        "damage": 50,
        "stamina_cost": 0.8,
        "time_cost": 0.6,
        "accuracy": 0.75,
        "points": 2
    },
    PunchType.REAR_HOOK: {
        "damage": 60,
        "stamina_cost": 1,
        "time_cost": 0.7,
        "accuracy": 0.7,
        "points": 3
    },
    PunchType.LEAD_UPPERCUT: {
        "damage": 70,
        "stamina_cost": 1.2,
        "time_cost": 0.8,
        "accuracy": 0.65,
        "points": 3
    },
    PunchType.REAR_UPPERCUT: {
        "damage": 80,
        "stamina_cost": 1.4,
        "time_cost": 0.9,
        "accuracy": 0.6,
        "points": 4
    }
}

@dataclass
class Boxer:
    name: str
    weight: float  # kg
    height: int    # cm
    reach: int     # cm
    power: int     # 1-100
    speed: int     # 1-100
    stamina: int   # 1-100
    defense: int   # 1-100
    chin: int      # 1-100
    experience: int # 1-100
    current_hp: float = 100.0
    current_stamina: float = 100.0
    
    @classmethod
    def generate(cls, name: str) -> 'Boxer':
        return cls(
            name=name,
            weight=random.uniform(60.0, 100.0),
            height=random.randint(165, 195),
            reach=random.randint(170, 200),
            power=random.randint(60, 100),
            speed=random.randint(60, 100),
            stamina=random.randint(60, 100),
            defense=random.randint(60, 100),
            chin=random.randint(60, 100),
            experience=random.randint(1, 100)
        )

@dataclass
class RoundState:
    knockdowns_A: int = 0
    knockdowns_B: int = 0
    points_A: int = 0
    points_B: int = 0
    clean_hits_A: int = 0
    clean_hits_B: int = 0
    total_punches_A: int = 0
    total_punches_B: int = 0
    # 新增的特徵
    stamina_used_A: float = 0.0
    stamina_used_B: float = 0.0
    damage_dealt_A: float = 0.0
    damage_dealt_B: float = 0.0
    punch_types_A: Dict[str, int] = None
    punch_types_B: Dict[str, int] = None
    
    def __post_init__(self):
        if self.punch_types_A is None:
            self.punch_types_A = {punch_type.value: 0 for punch_type in PunchType}
        if self.punch_types_B is None:
            self.punch_types_B = {punch_type.value: 0 for punch_type in PunchType}

class BoxingMatch:
    def __init__(self, boxerA: Boxer, boxerB: Boxer, rounds: int = 12):
        self.boxerA = boxerA
        self.boxerB = boxerB
        self.total_rounds = rounds
        self.current_round = 1
        self.round_state = RoundState()
        self.round_states: List[RoundState] = []
        self.detailed_actions: List[Dict] = []  # 記錄每個動作的詳細資訊

    def simulate_punch(self, attacker: Boxer, defender: Boxer, attacker_label: str, force_attack: bool = False) -> Dict:
        """模擬出拳並返回詳細資訊"""
        action_info = {
            "round": self.current_round,
            "attacker": attacker.name,
            "defender": defender.name,
            "attacker_hp": attacker.current_hp,
            "defender_hp": defender.current_hp,
            "attacker_stamina": attacker.current_stamina,
            "defender_stamina": defender.current_stamina,
            "is_defense": False,
            "punch_type": None,
            "is_hit": False,
            "damage": 0,
            "stamina_cost": 0,
            "is_knockdown": False
        }

        # 如果體力不足且不是強制攻擊，則進入防守狀態
        if attacker.current_stamina < 10 and not force_attack:
            stamina_recovered = random.uniform(5, 10)
            attacker.current_stamina = min(100, attacker.current_stamina + stamina_recovered)
            action_info["is_defense"] = True
            action_info["stamina_recovered"] = stamina_recovered
            return action_info

        punch_type = random.choice(list(PunchType))
        punch_info = PUNCH_PROPERTIES[punch_type]
        action_info["punch_type"] = punch_type.value

        if attacker_label == "A":
            self.round_state.total_punches_A += 1
            self.round_state.punch_types_A[punch_type.value] += 1
        else:
            self.round_state.total_punches_B += 1
            self.round_state.punch_types_B[punch_type.value] += 1

        hit_chance = (
            punch_info["accuracy"]
            * (attacker.speed / 100)
            * (attacker.current_stamina / 100)
            * (1 - defender.defense / 200)
        )

        is_hit = random.random() < hit_chance
        action_info["is_hit"] = is_hit

        if is_hit:
            damage = (
                punch_info["damage"]
                * (attacker.power / 100)
                * (attacker.current_stamina / 100)
                * (1 - defender.chin / 200)
            )
            
            defender.current_hp = max(0, defender.current_hp - damage)
            action_info["damage"] = damage

            if attacker_label == "A":
                self.round_state.clean_hits_A += 1
                self.round_state.points_A += punch_info["points"]
                self.round_state.damage_dealt_A += damage
            else:
                self.round_state.clean_hits_B += 1
                self.round_state.points_B += punch_info["points"]
                self.round_state.damage_dealt_B += damage

            if damage > 15 and random.random() < 0.15:
                action_info["is_knockdown"] = True
                if attacker_label == "A":
                    self.round_state.knockdowns_B += 1
                    defender.current_hp -= 10
                else:
                    self.round_state.knockdowns_A += 1
                    defender.current_hp -= 10

        stamina_cost = punch_info["stamina_cost"]
        attacker.current_stamina = max(0, attacker.current_stamina - stamina_cost)
        action_info["stamina_cost"] = stamina_cost

        if attacker_label == "A":
            self.round_state.stamina_used_A += stamina_cost
        else:
            self.round_state.stamina_used_B += stamina_cost

        return action_info

    def simulate_round(self) -> Tuple[bool, str, str]:
        """模擬一個回合"""
        time_left = 180.0  # 3分鐘
        self.round_state = RoundState()
        
        while time_left > 0:
            # A行動
            action_A = self.simulate_punch(self.boxerA, self.boxerB, "A")
            self.detailed_actions.append(action_A)
            
            # 檢查是否KO
            if self.boxerB.current_hp <= 0:
                self.round_states.append(self.round_state)  # 確保保存最後一回合的數據
                return True, self.boxerA.name, "KO"

            # B行動
            action_B = self.simulate_punch(self.boxerB, self.boxerA, "B")
            self.detailed_actions.append(action_B)
            
            # 檢查是否KO
            if self.boxerA.current_hp <= 0:
                self.round_states.append(self.round_state)  # 確保保存最後一回合的數據
                return True, self.boxerB.name, "KO"

            time_left -= 2

        self._end_round_processing()
        self.round_states.append(self.round_state)
        return False, "", ""

    def _end_round_processing(self):
        """回合結束處理"""
        for boxer in [self.boxerA, self.boxerB]:
            boxer.current_stamina = min(100, boxer.current_stamina + 30)
            boxer.current_hp = min(100, boxer.current_hp + 10)

    def _judge_match(self) -> Tuple[str, str]:
        """判定比賽結果"""
        total_points_A = sum(r.points_A for r in self.round_states)
        total_points_B = sum(r.points_B for r in self.round_states)
        
        point_diff = abs(total_points_A - total_points_B)
        if point_diff < 10:
            if total_points_A > total_points_B:
                return self.boxerA.name, "Split Decision"
            elif total_points_B > total_points_A:
                return self.boxerB.name, "Split Decision"
            else:
                return "Draw", "Draw"
        else:
            if total_points_A > total_points_B:
                return self.boxerA.name, "Unanimous Decision"
            else:
                return self.boxerB.name, "Unanimous Decision"

    def simulate_match(self) -> Dict:
        """模擬整場比賽"""
        match_ended = False
        winner = ""
        victory_condition = ""
        
        while self.current_round <= self.total_rounds and not match_ended:
            match_ended, winner, victory_condition = self.simulate_round()
            if not match_ended:
                self.current_round += 1
        
        if not match_ended:
            winner, victory_condition = self._judge_match()
        
        return self._create_match_result(winner, victory_condition)

    def _create_match_result(self, winner: str, victory_condition: str) -> Dict:
        """創建詳細的比賽結果數據"""
        # 基本比賽資訊
        result = {
            "match_id": random.randint(10000, 99999),
            "winner": winner,
            "victory_condition": victory_condition,
            "rounds_completed": self.current_round,
            
            # 選手A基本資料
            "boxerA_name": self.boxerA.name,
            "boxerA_weight": round(self.boxerA.weight, 1),
            "boxerA_height": self.boxerA.height,
            "boxerA_reach": self.boxerA.reach,
            "boxerA_power": self.boxerA.power,
            "boxerA_speed": self.boxerA.speed,
            "boxerA_stamina": self.boxerA.stamina,
            "boxerA_defense": self.boxerA.defense,
            "boxerA_chin": self.boxerA.chin,
            "boxerA_experience": self.boxerA.experience,
            
            # 選手B基本資料
            "boxerB_name": self.boxerB.name,
            "boxerB_weight": round(self.boxerB.weight, 1),
            "boxerB_height": self.boxerB.height,
            "boxerB_reach": self.boxerB.reach,
            "boxerB_power": self.boxerB.power,
            "boxerB_speed": self.boxerB.speed,
            "boxerB_stamina": self.boxerB.stamina,
            "boxerB_defense": self.boxerB.defense,
            "boxerB_chin": self.boxerB.chin,
            "boxerB_experience": self.boxerB.experience,
        }
        
        # 計算整場比賽的總計數據
        total_stats = {
            "total_punches_A": sum(r.total_punches_A for r in self.round_states),
            "total_punches_B": sum(r.total_punches_B for r in self.round_states),
            "total_hits_A": sum(r.clean_hits_A for r in self.round_states),
            "total_hits_B": sum(r.clean_hits_B for r in self.round_states),
            "total_knockdowns_A": sum(r.knockdowns_A for r in self.round_states),
            "total_knockdowns_B": sum(r.knockdowns_B for r in self.round_states),
            "total_points_A": sum(r.points_A for r in self.round_states),
            "total_points_B": sum(r.points_B for r in self.round_states),
            "total_damage_dealt_A": sum(r.damage_dealt_A for r in self.round_states),
            "total_damage_dealt_B": sum(r.damage_dealt_B for r in self.round_states),
            "total_stamina_used_A": sum(r.stamina_used_A for r in self.round_states),
            "total_stamina_used_B": sum(r.stamina_used_B for r in self.round_states),
        }
        
        # 計算每種拳的總使用次數
        punch_counts_A = {punch_type.value: 0 for punch_type in PunchType}
        punch_counts_B = {punch_type.value: 0 for punch_type in PunchType}
        
        for round_state in self.round_states:
            for punch_type, count in round_state.punch_types_A.items():
                punch_counts_A[punch_type] += count
            for punch_type, count in round_state.punch_types_B.items():
                punch_counts_B[punch_type] += count
        
        # 添加總計數據到結果
        result.update(total_stats)
        
        # 添加命中率
        result["hit_rate_A"] = round(total_stats["total_hits_A"] / total_stats["total_punches_A"] * 100, 2) if total_stats["total_punches_A"] > 0 else 0
        result["hit_rate_B"] = round(total_stats["total_hits_B"] / total_stats["total_punches_B"] * 100, 2) if total_stats["total_punches_B"] > 0 else 0
        
        # 添加最終狀態
        result.update({
            "final_hp_A": round(self.boxerA.current_hp, 2),
            "final_hp_B": round(self.boxerB.current_hp, 2),
            "final_stamina_A": round(self.boxerA.current_stamina, 2),
            "final_stamina_B": round(self.boxerB.current_stamina, 2),
        })
        
        # 添加每種拳的總使用次數
        for punch_type in PunchType:
            punch_name = punch_type.value.lower().replace(' ', '_')
            result[f"total_A_{punch_name}"] = punch_counts_A[punch_type.value]
            result[f"total_B_{punch_name}"] = punch_counts_B[punch_type.value]
        
        return result

def simulate_matches(num_matches: int = 1000, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """模擬多場拳擊比賽並返回DataFrame"""
    results = []
    
    for i in range(num_matches):
        if verbose and i % 100 == 0:
            print(f"模擬第 {i+1}/{num_matches} 場比賽...")
            
        boxerA = Boxer.generate(f"Boxer_A_{i+1}")
        boxerB = Boxer.generate(f"Boxer_B_{i+1}")
        
        match = BoxingMatch(boxerA, boxerB, rounds=12)
        result = match.simulate_match()
        
        # 驗證數據完整性
        if not any(result[f"total_{k}"] == 0 for k in ["punches_A", "punches_B"]):
            results.append(result)
        elif verbose:
            print(f"警告: 第 {i+1} 場比賽的數據不完整，重新模擬...")
    
    df = pd.DataFrame(results)
    
    if save_path:
        output_dir = Path(save_path).parent
        output_dir.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"\n數據已保存到: {save_path}")
        print("\n基本統計:")
        print(f"總場次: {num_matches}")
        print(f"KO率: {(df['victory_condition'] == 'KO').mean()*100:.2f}%")
        print(f"判決勝率: {(df['victory_condition'].str.contains('Decision')).mean()*100:.2f}%")
        print(f"平均回合數: {df['rounds_completed'].mean():.2f}")
        print(f"平均命中率A: {df['hit_rate_A'].mean():.2f}%")
        print(f"平均命中率B: {df['hit_rate_B'].mean():.2f}%")
    
    return df

if __name__ == "__main__":
    output_path = "boxing-matches.csv"
    df = simulate_matches(10000, output_path)
    
    # 顯示基本統計
    print("\n比賽統計:")
    print(f"KO率: {(df['victory_condition'] == 'KO').mean()*100:.2f}%")
    print(f"TKO率: {(df['victory_condition'] == 'TKO').mean()*100:.2f}%")
    print(f"判決勝率: {(df['victory_condition'].str.contains('Decision')).mean()*100:.2f}%")
    print(f"平均回合數: {df['rounds_completed'].mean():.2f}")
    print(f"平均命中率A: {df['hit_rate_A'].mean():.2f}%")
    print(f"平均命中率B: {df['hit_rate_B'].mean():.2f}%")

    # count = 0
    # # 創建比賽實例
    # for i in range(1000):
    #     # 手動創建拳擊手
    #     tyson = Boxer(
    #         name="Mike Tyson",
    #         weight=95.5,    # 體重(kg)
    #         height=178,     # 身高(cm)
    #         reach=180,      # 臂展(cm)
    #         power=95,       # 力量(1-100)
    #         speed=90,       # 速度(1-100)
    #         stamina=85,     # 體力(1-100)
    #         defense=80,     # 防守(1-100)
    #         chin=85,        # 抗打擊能力(1-100)
    #         experience=95   # 經驗(1-100)
    #     )

    #     ali = Boxer(
    #         name="Muhammad Ali",
    #         weight=92.0,
    #         height=191,
    #         reach=198,
    #         power=88,
    #         speed=95,
    #         stamina=95,
    #         defense=90,
    #         chin=90,
    #         experience=98
    #     )
    #     match = BoxingMatch(tyson, ali, rounds=12)

    #     # 模擬一場比賽
    #     result = match.simulate_match()
        
    #     # print(result["winner"])
    #     if result['winner'] == "Mike Tyson":
    #         count += 1

    # print(f"Mike Tyson wins {count} in 1000 games")

    # # 印出結果
    # # print(f"\n比賽結果:")
    # # print(f"獲勝者: {result['winner']}")
    # # print(f"獲勝方式: {result['victory_condition']}")
    # # print(f"完成回合數: {result['rounds_completed']}")