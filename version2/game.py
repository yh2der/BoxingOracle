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

# 加入拳種特性
PUNCH_HEIGHT_PREFERENCE = {
    PunchType.JAB: 0,          # 中性
    PunchType.CROSS: 0,        # 中性
    PunchType.LEAD_HOOK: 0,    # 中性
    PunchType.REAR_HOOK: 0,    # 中性
    PunchType.LEAD_UPPERCUT: -0.002,  # 矮打高優勢
    PunchType.REAR_UPPERCUT: -0.002   # 矮打高優勢
}

PUNCH_REACH_IMPORTANCE = {
    PunchType.JAB: 1.2,        # 臂展影響更大
    PunchType.CROSS: 1.2,      # 臂展影響更大
    PunchType.LEAD_HOOK: 0.8,  # 臂展影響較小
    PunchType.REAR_HOOK: 0.8,  # 臂展影響較小
    PunchType.LEAD_UPPERCUT: 0.6,  # 臂展影響最小
    PunchType.REAR_UPPERCUT: 0.6   # 臂展影響最小
}

PUNCH_WEIGHT_IMPORTANCE = {
    PunchType.JAB: 0.6,        # 體重影響較小
    PunchType.CROSS: 0.8,      # 體重影響一般
    PunchType.LEAD_HOOK: 1.2,  # 體重影響較大
    PunchType.REAR_HOOK: 1.2,  # 體重影響較大
    PunchType.LEAD_UPPERCUT: 1.0,  # 體重影響一般
    PunchType.REAR_UPPERCUT: 1.0   # 體重影響一般
}

PUNCH_PROPERTIES = {
    PunchType.JAB: {
        "damage": 6,
        "stamina_cost": 0.3,
        "time_cost": 0.3,
        "accuracy": 0.85,
        "points": 1
    },
    PunchType.CROSS: {
        "damage": 11,
        "stamina_cost": 0.6,
        "time_cost": 0.5,
        "accuracy": 0.75,
        "points": 2
    },
    PunchType.LEAD_HOOK: {
        "damage": 15,
        "stamina_cost": 0.8,
        "time_cost": 0.6,
        "accuracy": 0.7,
        "points": 2
    },
    PunchType.REAR_HOOK: {
        "damage": 18,
        "stamina_cost": 1,
        "time_cost": 0.7,
        "accuracy": 0.65,
        "points": 3
    },
    PunchType.LEAD_UPPERCUT: {
        "damage": 20,
        "stamina_cost": 1.2,
        "time_cost": 0.8,
        "accuracy": 0.6,
        "points": 3
    },
    PunchType.REAR_UPPERCUT: {
        "damage": 24,
        "stamina_cost": 1.4,
        "time_cost": 0.9,
        "accuracy": 0.55,
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
            weight=random.uniform(70.0, 90.0),
            height=random.randint(170, 185),
            reach=random.randint(170, 185),
            power=random.randint(70, 90),
            speed=random.randint(70, 90),
            stamina=random.randint(70, 90),
            defense=random.randint(70, 90),
            chin=random.randint(70, 90),
            experience=random.randint(50, 100)
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
        self.detailed_actions: List[Dict] = []

    def calculate_physical_advantages(self, attacker: Boxer, defender: Boxer, punch_type: PunchType) -> Tuple[float, float]:
        """計算基於物理屬性的優勢"""
        # 1. 臂展優勢 (每1cm差異影響0.1%，受拳種影響)
        reach_advantage = ((attacker.reach - defender.reach) * 0.001 
                         * PUNCH_REACH_IMPORTANCE[punch_type])
        
        # 2. 身高優勢 (每1cm差異影響0.2%，uppercut有特殊計算)
        height_difference = attacker.height - defender.height
        height_advantage = (height_difference * PUNCH_HEIGHT_PREFERENCE[punch_type])
        
        # 3. 體重優勢 (每1kg差異影響0.5%，受拳種影響)
        weight_advantage = ((attacker.weight - defender.weight) * 0.005 
                          * PUNCH_WEIGHT_IMPORTANCE[punch_type])
        
        # 分別返回命中率修正和傷害修正
        hit_modifier = 1 + reach_advantage + height_advantage
        damage_modifier = 1 + weight_advantage
        
        return hit_modifier, damage_modifier

    def simulate_punch(self, attacker: Boxer, defender: Boxer, attacker_label: str) -> Dict:
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

        if attacker.current_stamina < 10:
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

        # 計算物理優勢
        hit_modifier, damage_modifier = self.calculate_physical_advantages(attacker, defender, punch_type)

        # 命中率計算 (加入物理優勢)
        hit_chance = (
            punch_info["accuracy"]
            * (attacker.speed / 100)
            * (attacker.current_stamina / 100)
            * (1 - defender.defense / 200)
            * hit_modifier  # 加入物理優勢影響
        )
        is_hit = random.random() < hit_chance
        action_info["is_hit"] = is_hit

        if is_hit:
            # 傷害計算 (加入物理優勢)
            damage = (
                punch_info["damage"]
                * (attacker.power / 100)
                * (attacker.current_stamina / 100)
                * (1 - defender.chin / 220)
                * 0.80
                * damage_modifier  # 加入物理優勢影響
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

            knockout_chance = (
                0.005
                * (damage / 18)
                * (1 - defender.current_hp / 100)
                * (attacker.power / 100)
                * (1 - defender.chin / 140)
                * (1 - defender.current_stamina / 180)
                * damage_modifier  # 加入物理優勢對擊倒的影響
            )
            
            if random.random() < knockout_chance:
                action_info["is_knockdown"] = True
                if attacker_label == "A":
                    self.round_state.knockdowns_B += 1
                    defender.current_hp -= 12
                else:
                    self.round_state.knockdowns_A += 1
                    defender.current_hp -= 12

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
        time_left = 180.0
        self.round_state = RoundState()
        
        while time_left > 0:
            action_A = self.simulate_punch(self.boxerA, self.boxerB, "A")
            self.detailed_actions.append(action_A)
            
            if self.boxerB.current_hp <= 0:
                self.round_states.append(self.round_state)
                return True, self.boxerA.name, "KO"

            action_B = self.simulate_punch(self.boxerB, self.boxerA, "B")
            self.detailed_actions.append(action_B)
            
            if self.boxerA.current_hp <= 0:
                self.round_states.append(self.round_state)
                return True, self.boxerB.name, "KO"

            time_left -= 2

        self._end_round_processing()
        self.round_states.append(self.round_state)
        return False, "", ""

    def _end_round_processing(self):
        """回合結束處理"""
        for boxer in [self.boxerA, self.boxerB]:
            boxer.current_stamina = min(100, boxer.current_stamina + 20)
            boxer.current_hp = min(100, boxer.current_hp + 7)

    def _judge_match(self) -> Tuple[str, str]:
        """判定比賽結果"""
        total_points_A = sum(r.points_A for r in self.round_states)
        total_points_B = sum(r.points_B for r in self.round_states)
        point_diff = abs(total_points_A - total_points_B)

        if point_diff < 5:
            return "Draw", "Draw"
        elif point_diff < 40:
            if total_points_A > total_points_B:
                return self.boxerA.name, "Split Decision"
            else:
                return self.boxerB.name, "Split Decision"
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
    # output_path = "boxing-matches.csv"
    # df = simulate_matches(10000, output_path)

    
    # (B) 自訂拳手對戰範例
    # =========================
    tyson = Boxer(
        name="Iron Mike Tyson",
        weight=95.5,
        height=178,
        reach=180,
        power=95,
        speed=90,
        stamina=85,
        defense=80,
        chin=85,
        experience=95
    )

    ali = Boxer(
        name="Muhammad Ali",
        weight=92.0,
        height=191,
        reach=198,
        power=88,
        speed=95,
        stamina=95,
        defense=90,
        chin=90,
        experience=98
    )

    # 建立比賽
    match = BoxingMatch(tyson, ali, rounds=1)

    # 開始模擬比賽
    result = match.simulate_match()

    # 只想印出誰贏
    print("Winner:", result["winner"])

    # 若想保存成 CSV
    # df_custom.to_csv("tyson_vs_ali.csv", index=False)
    # print("\n已將 Tyson vs Ali 的對戰結果存為 tyson_vs_ali.csv")