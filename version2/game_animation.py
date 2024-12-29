import random
import time
import os
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from pathlib import Path
from colorama import init, Fore, Back, Style

init(autoreset=True)

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
    chin: int      # 1-100 (承受打擊能力)
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

class LiveBoxingMatch:
    def __init__(self, boxerA: Boxer, boxerB: Boxer, rounds: int = 12):
        self.boxerA = boxerA
        self.boxerB = boxerB
        self.total_rounds = rounds
        self.current_round = 1
        self.ring_width = 80
        self.round_state = RoundState()
        self.round_states: List[RoundState] = []
        
    def display_stats(self, time_left: float):
        """顯示比賽狀態"""
        self.clear_screen()
        print(Fore.CYAN + f"\n{'='*self.ring_width}")
        print(Fore.YELLOW + f"第 {self.current_round} 回合    剩餘時間: {time_left:.1f} 秒")
        print(Fore.CYAN + f"{'='*self.ring_width}\n")
        
        # 顯示選手數據
        self._display_boxer_stats(self.boxerA, "left")
        print()
        self._display_boxer_stats(self.boxerB, "right")
        
        # 顯示回合數據
        print(Fore.CYAN + f"\n{'='*self.ring_width}")
        print(Fore.WHITE + f"本回合數據: A命中率: {self._calculate_hit_rate('A'):.1f}% | " + 
              f"B命中率: {self._calculate_hit_rate('B'):.1f}%")
        print(Fore.WHITE + f"          得分: A {self.round_state.points_A} | B {self.round_state.points_B}")
        print(Fore.CYAN + f"{'='*self.ring_width}")
    
    def _display_boxer_stats(self, boxer: Boxer, align: str):
        """顯示選手狀態"""
        bar_length = 30
        hp_filled = int((boxer.current_hp / 100) * bar_length)
        stamina_filled = int((boxer.current_stamina / 100) * bar_length)
        
        hp_bar = f"[{'■'*hp_filled}{'□'*(bar_length-hp_filled)}]"
        stamina_bar = f"[{'■'*stamina_filled}{'□'*(bar_length-stamina_filled)}]"
        
        stats = (f"{boxer.name} "
                f"(力量:{boxer.power} 速度:{boxer.speed} "
                f"體力:{boxer.stamina} 防禦:{boxer.defense})")
        
        if align == "left":
            print(f"{Fore.GREEN}{stats:<60}")
            print(f"HP:  {hp_bar} {boxer.current_hp:4.1f}%")
            print(f"體力: {stamina_bar} {boxer.current_stamina:4.1f}%")
        else:
            print(f"{Fore.GREEN}{stats:<60}")
            print(f"HP:  {hp_bar} {boxer.current_hp:4.1f}%")
            print(f"體力: {stamina_bar} {boxer.current_stamina:4.1f}%")

    def _calculate_hit_rate(self, boxer_label: str) -> float:
        """計算命中率"""
        hits = (self.round_state.clean_hits_A if boxer_label == "A" 
               else self.round_state.clean_hits_B)
        punches = (self.round_state.total_punches_A if boxer_label == "A" 
                  else self.round_state.total_punches_B)
        return (hits / punches * 100) if punches > 0 else 0

    def simulate_punch(self, attacker: Boxer, defender: Boxer, attacker_label: str) -> Tuple[bool, str]:
        """
        模擬出拳或防禦
        返回值:
            is_defense: 是否防禦
            action_result: 行為結果描述
        """
        # 如果攻擊者的體力不足，進入防禦模式
        if attacker.current_stamina < 10:
            # 恢復體力，消耗回合時間
            stamina_recovered = random.uniform(5, 10)
            attacker.current_stamina = min(100, attacker.current_stamina + stamina_recovered)
            return True, f"{attacker.name} 防禦並恢復了 {stamina_recovered:.1f} 體力"
        
        # 選擇拳種
        punch_type = random.choice(list(PunchType))
        punch_info = PUNCH_PROPERTIES[punch_type]

        # 更新總出拳數
        if attacker_label == "A":
            self.round_state.total_punches_A += 1
        else:
            self.round_state.total_punches_B += 1

        # 命中判定
        hit_chance = (
            punch_info["accuracy"]
            * (attacker.speed / 100)
            * (attacker.current_stamina / 100)
            * (1 - defender.defense / 200)
        )

        is_hit = random.random() < hit_chance
        is_knockdown = False

        if is_hit:
            # 計算傷害
            damage = (
                punch_info["damage"]
                * (attacker.power / 100)
                * (attacker.current_stamina / 100)
                * (1 - defender.chin / 200)
            )

            # 更新對手的HP
            defender.current_hp = max(0, defender.current_hp - damage)

            # 更新得分
            if attacker_label == "A":
                self.round_state.clean_hits_A += 1
                self.round_state.points_A += punch_info["points"]
            else:
                self.round_state.clean_hits_B += 1
                self.round_state.points_B += punch_info["points"]

            # 擊倒判定
            if damage > 15 and random.random() < 0.15:
                is_knockdown = True
                if attacker_label == "A":
                    self.round_state.knockdowns_B += 1
                    defender.current_hp -= 10
                else:
                    self.round_state.knockdowns_A += 1
                    defender.current_hp -= 10

        # 消耗體力
        attacker.current_stamina = max(0, attacker.current_stamina - punch_info["stamina_cost"])

        # 返回出拳結果
        if is_hit:
            if is_knockdown:
                return False, f"{attacker.name} 使用 {punch_type.value}，造成 KNOCKDOWN!"
            else:
                return False, f"{attacker.name} 使用 {punch_type.value}，造成 {damage:.1f} 傷害!"
        else:
            return False, f"{attacker.name} 使用 {punch_type.value}，但未命中!"


    def display_action(self, attacker: Boxer, punch_type: PunchType, damage: float, 
                  is_hit: bool, is_knockdown: bool, position: str):
        """顯示戰鬥動作"""
        action = f"{attacker.name} 使用 {punch_type.value}"
        if is_hit:
            if is_knockdown:
                effect = "===[ KNOCKDOWN! ]==="
            else:
                effect = self.get_hit_effect(damage)
            
            # 兩邊都使用靠左對齊
            print(f"{Fore.YELLOW}{action:30} {Fore.RED}{effect:>15} {damage:4.1f}")
        else:
            miss_text = "[ MISS! ]"
            # 兩邊都使用靠左對齊
            print(f"{Fore.YELLOW}{action:30} {Fore.RED}{miss_text:>15}")

    def get_hit_effect(self, damage: float) -> str:
        """根據傷害程度返回打擊特效"""
        if damage >= 20:
            return "===[ BOOM! ]==="
        elif damage >= 15:
            return "==[ POW! ]=="
        elif damage >= 10:
            return "=[ BAM! ]="
        else:
            return "[ hit ]"

    def simulate_round(self) -> Tuple[bool, str, str]:
        """模擬一個回合"""
        time_left = 180.0  # 3分鐘
        self.round_state = RoundState()

        while time_left > 0:
            self.display_stats(time_left)

            # A行動
            is_defense, action_result = self.simulate_punch(self.boxerA, self.boxerB, "A")
            print(Fore.YELLOW + action_result)
            if not is_defense and self.boxerB.current_hp <= 0:
                time.sleep(2)
                return True, self.boxerA.name, "KO"

            time.sleep(0.5)

            # B行動
            is_defense, action_result = self.simulate_punch(self.boxerB, self.boxerA, "B")
            print(Fore.YELLOW + action_result)
            if not is_defense and self.boxerA.current_hp <= 0:
                time.sleep(2)
                return True, self.boxerB.name, "KO"

            time_left -= 2
            time.sleep(0.5)

        # 回合結束，恢復體力
        self._end_round_processing()
        self.round_states.append(self.round_state)
        return False, "", ""


    def _end_round_processing(self):
        """回合结束处理"""
        for boxer in [self.boxerA, self.boxerB]:
            boxer.current_stamina = min(100, boxer.current_stamina + 30)  # 恢复更多体力
            boxer.current_hp = min(100, boxer.current_hp + 10)           # 恢复更多生命值


    def simulate_match(self) -> Dict:
        """模擬整場比賽"""
        print(Fore.MAGENTA + f"\n{'='*self.ring_width}")
        print(Fore.YELLOW + f"拳擊比賽: {self.boxerA.name} vs {self.boxerB.name}")
        print(Fore.MAGENTA + f"{'='*self.ring_width}")
        time.sleep(2)
        
        match_ended = False
        winner = ""
        victory_condition = ""
        
        while self.current_round <= self.total_rounds and not match_ended:
            match_ended, winner, victory_condition = self.simulate_round()
            if not match_ended:
                self.current_round += 1
        
        if not match_ended:
            winner, victory_condition = self._judge_match()
        
        self.display_match_result(winner, victory_condition)
        return self._create_match_result(winner, victory_condition)

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
    
    def display_match_result(self, winner: str, victory_condition: str):
        """顯示比賽結果"""
        self.clear_screen()
        print(Fore.CYAN + f"\n{'='*self.ring_width}")
        print(Fore.YELLOW + "比賽結束!")
        print(Fore.CYAN + f"{'='*self.ring_width}")
        
        total_points_A = sum(r.points_A for r in self.round_states)
        total_points_B = sum(r.points_B for r in self.round_states)
        total_hits_A = sum(r.clean_hits_A for r in self.round_states)
        total_hits_B = sum(r.clean_hits_B for r in self.round_states)
        
        print(f"\n{self.boxerA.name} 最終數據:")
        print(f"HP: {self.boxerA.current_hp:.1f}%")
        print(f"總得分: {total_points_A}")
        print(f"命中數: {total_hits_A}")
        
        print(f"\n{self.boxerB.name} 最終數據:")
        print(f"HP: {self.boxerB.current_hp:.1f}%")
        print(f"總得分: {total_points_B}")
        print(f"命中數: {total_hits_B}")
        
        print(Fore.CYAN + f"\n{'='*self.ring_width}")
        if winner == "Draw":
            print(Fore.YELLOW + f"結果: 比賽平手! ({victory_condition})")
        else:
            print(Fore.YELLOW + f"勝利者: {winner} ({victory_condition})")
        print(Fore.CYAN + f"{'='*self.ring_width}\n")
    
    def _create_match_result(self, winner: str, victory_condition: str) -> Dict:
        """創建比賽結果數據"""
        total_punches_A = sum(r.total_punches_A for r in self.round_states)
        total_punches_B = sum(r.total_punches_B for r in self.round_states)
        total_hits_A = sum(r.clean_hits_A for r in self.round_states)
        total_hits_B = sum(r.clean_hits_B for r in self.round_states)
        total_knockdowns_A = sum(r.knockdowns_A for r in self.round_states)
        total_knockdowns_B = sum(r.knockdowns_B for r in self.round_states)
        
        return {
            "boxerA_name": self.boxerA.name,
            "boxerB_name": self.boxerB.name,
            "boxerA_weight": round(self.boxerA.weight, 1),
            "boxerB_weight": round(self.boxerB.weight, 1),
            "boxerA_height": self.boxerA.height,
            "boxerB_height": self.boxerB.height,
            "boxerA_reach": self.boxerA.reach,
            "boxerB_reach": self.boxerB.reach,
            "boxerA_power": self.boxerA.power,
            "boxerB_power": self.boxerB.power,
            "boxerA_speed": self.boxerA.speed,
            "boxerB_speed": self.boxerB.speed,
            "boxerA_stamina": self.boxerA.stamina,
            "boxerB_stamina": self.boxerB.stamina,
            "boxerA_defense": self.boxerA.defense,
            "boxerB_defense": self.boxerB.defense,
            "boxerA_chin": self.boxerA.chin,
            "boxerB_chin": self.boxerB.chin,
            "boxerA_experience": self.boxerA.experience,
            "boxerB_experience": self.boxerB.experience,
            "rounds_completed": self.current_round,
            "final_hp_A": round(self.boxerA.current_hp, 2),
            "final_hp_B": round(self.boxerB.current_hp, 2),
            "total_punches_A": total_punches_A,
            "total_punches_B": total_punches_B,
            "total_hits_A": total_hits_A,
            "total_hits_B": total_hits_B,
            "total_knockdowns_A": total_knockdowns_A,
            "total_knockdowns_B": total_knockdowns_B,
            "hit_rate_A": round(total_hits_A / total_punches_A * 100, 2) if total_punches_A > 0 else 0,
            "hit_rate_B": round(total_hits_B / total_punches_B * 100, 2) if total_punches_B > 0 else 0,
            "winner": winner,
            "victory_condition": victory_condition
        }
    
    @staticmethod
    def clear_screen():
        """清除螢幕"""
        os.system('cls' if os.name == 'nt' else 'clear')

def simulate_matches(num_matches: int = 1000) -> None:
    """模擬多場拳擊比賽並保存結果"""
    results = []
    
    for i in range(num_matches):
        print(f"模擬第 {i+1}/{num_matches} 場比賽...")
        
        boxerA = Boxer.generate(f"Boxer_A_{i+1}")
        boxerB = Boxer.generate(f"Boxer_B_{i+1}")
        
        match = LiveBoxingMatch(boxerA, boxerB, rounds=12)
        result = match.simulate_match()
        results.append(result)
        
        if i < num_matches - 1:  # 不是最後一場才暫停
            input("\n按Enter繼續下一場比賽...")
    
    # 保存和分析數據
    df = pd.DataFrame(results)
    output_dir = Path("boxing_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"boxing_matches_{num_matches}.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n數據已保存到: {output_path}")
    print("\n比賽統計:")
    print(f"總場次: {num_matches}")
    print(f"KO率: {(df['victory_condition'] == 'KO').mean()*100:.2f}%")
    print(f"判決勝率: {(df['victory_condition'].str.contains('Decision')).mean()*100:.2f}%")
    print(f"平均回合數: {df['rounds_completed'].mean():.2f}")
    print(f"平均命中率A: {df['hit_rate_A'].mean():.2f}%")
    print(f"平均命中率B: {df['hit_rate_B'].mean():.2f}%")

if __name__ == "__main__":    
    # 示範多場比賽
    #simulate_matches(5)  # 模擬5場比賽

    # 建立自定義拳擊手
    tyson = Boxer(
        name="Iron Mike Tyson",
        weight=95.5,    # 體重(kg)
        height=178,     # 身高(cm)
        reach=180,      # 臂展(cm)
        power=95,       # 力量(1-100)
        speed=90,       # 速度(1-100)
        stamina=85,     # 體力(1-100)
        defense=80,     # 防守(1-100)
        chin=85,        # 抗打擊能力(1-100)
        experience=95   # 經驗(1-100)
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
    match = LiveBoxingMatch(tyson, ali, rounds=12)

    # 開始模擬比賽
    result = match.simulate_match()