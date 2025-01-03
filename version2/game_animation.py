import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from colorama import Fore, Style, init
from dataclasses import dataclass
from enum import Enum

# 初始化 colorama
init(autoreset=True)

# ------------------------------------------------------------
#  拳種定義與特性
# ------------------------------------------------------------
class PunchType(Enum):
    JAB = "Jab"
    CROSS = "Cross"
    LEAD_HOOK = "Lead Hook"
    REAR_HOOK = "Rear Hook"
    LEAD_UPPERCUT = "Lead Uppercut"
    REAR_UPPERCUT = "Rear Uppercut"


# 加入拳種特性 (身高、體重、臂展對出拳效果的影響)
PUNCH_HEIGHT_PREFERENCE = {
    PunchType.JAB: 0,
    PunchType.CROSS: 0,
    PunchType.LEAD_HOOK: 0,
    PunchType.REAR_HOOK: 0,
    PunchType.LEAD_UPPERCUT: -0.002,  # Uppercut矮個較有利
    PunchType.REAR_UPPERCUT: -0.002,
}

PUNCH_REACH_IMPORTANCE = {
    PunchType.JAB: 1.2,
    PunchType.CROSS: 1.2,
    PunchType.LEAD_HOOK: 0.8,
    PunchType.REAR_HOOK: 0.8,
    PunchType.LEAD_UPPERCUT: 0.6,
    PunchType.REAR_UPPERCUT: 0.6,
}

PUNCH_WEIGHT_IMPORTANCE = {
    PunchType.JAB: 0.6,
    PunchType.CROSS: 0.8,
    PunchType.LEAD_HOOK: 1.2,
    PunchType.REAR_HOOK: 1.2,
    PunchType.LEAD_UPPERCUT: 1.0,
    PunchType.REAR_UPPERCUT: 1.0,
}

# 拳種基礎屬性
PUNCH_PROPERTIES = {
    PunchType.JAB: {
        "damage": 6,
        "stamina_cost": 0.3,
        "time_cost": 0.3,
        "accuracy": 0.85,
        "points": 1,
    },
    PunchType.CROSS: {
        "damage": 11,
        "stamina_cost": 0.6,
        "time_cost": 0.5,
        "accuracy": 0.75,
        "points": 2,
    },
    PunchType.LEAD_HOOK: {
        "damage": 15,
        "stamina_cost": 0.8,
        "time_cost": 0.6,
        "accuracy": 0.7,
        "points": 2,
    },
    PunchType.REAR_HOOK: {
        "damage": 18,
        "stamina_cost": 1,
        "time_cost": 0.7,
        "accuracy": 0.65,
        "points": 3,
    },
    PunchType.LEAD_UPPERCUT: {
        "damage": 20,
        "stamina_cost": 1.2,
        "time_cost": 0.8,
        "accuracy": 0.6,
        "points": 3,
    },
    PunchType.REAR_UPPERCUT: {
        "damage": 24,
        "stamina_cost": 1.4,
        "time_cost": 0.9,
        "accuracy": 0.55,
        "points": 4,
    },
}


# ------------------------------------------------------------
#  賽事相關資料結構
# ------------------------------------------------------------
@dataclass
class Boxer:
    name: str
    weight: float  # kg
    height: int  # cm
    reach: int  # cm
    power: int  # 1-100
    speed: int  # 1-100
    stamina: int  # 1-100
    defense: int  # 1-100
    chin: int  # 1-100
    experience: int  # 1-100
    current_hp: float = 100.0
    current_stamina: float = 100.0

    @classmethod
    def generate(cls, name: str) -> "Boxer":
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
            experience=random.randint(50, 100),
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
            self.punch_types_A = {p.value: 0 for p in PunchType}
        if self.punch_types_B is None:
            self.punch_types_B = {p.value: 0 for p in PunchType}


# ------------------------------------------------------------
#  模擬拳擊比賽 (含動畫效果)
# ------------------------------------------------------------
class LiveBoxingMatch:
    def __init__(self, boxerA: Boxer, boxerB: Boxer, rounds: int = 12):
        self.boxerA = boxerA
        self.boxerB = boxerB
        self.total_rounds = rounds
        self.current_round = 1
        self.round_state = RoundState()
        self.round_states: List[RoundState] = []
        self.ring_width = 80  # 顯示用

    def clear_screen(self):
        """清除螢幕"""
        os.system("cls" if os.name == "nt" else "clear")

    def display_stats(self, time_left: float):
        """顯示回合中雙方即時狀態 (動畫效果)"""
        self.clear_screen()
        print(Fore.CYAN + f"\n{'=' * self.ring_width}")
        print(Fore.YELLOW + f"第 {self.current_round} 回合    剩餘時間: {time_left:.1f} 秒")
        print(Fore.CYAN + f"{'=' * self.ring_width}\n")

        # 顯示選手 A 狀態
        self._display_boxer_stats(self.boxerA, "A")
        print()
        # 顯示選手 B 狀態
        self._display_boxer_stats(self.boxerB, "B")

        # 顯示回合中得分/命中資訊
        print(Fore.CYAN + f"\n{'=' * self.ring_width}")
        hit_rate_A = (
            (self.round_state.clean_hits_A / self.round_state.total_punches_A * 100)
            if self.round_state.total_punches_A > 0
            else 0
        )
        hit_rate_B = (
            (self.round_state.clean_hits_B / self.round_state.total_punches_B * 100)
            if self.round_state.total_punches_B > 0
            else 0
        )
        print(
            Fore.WHITE
            + f"本回合 A 命中率: {hit_rate_A:.1f}%"
            f" | B 命中率: {hit_rate_B:.1f}%"
        )
        print(
            Fore.WHITE
            + f"          得分: A {self.round_state.points_A} | B {self.round_state.points_B}"
        )
        print(Fore.CYAN + f"{'=' * self.ring_width}")

    def _display_boxer_stats(self, boxer: Boxer, label: str):
        """顯示單位行的血條、體力條"""
        bar_length = 30
        hp_filled = int((boxer.current_hp / 100) * bar_length)
        st_filled = int((boxer.current_stamina / 100) * bar_length)

        hp_bar = f"[{'■' * hp_filled}{'□' * (bar_length - hp_filled)}]"
        st_bar = f"[{'■' * st_filled}{'□' * (bar_length - st_filled)}]"

        stats = (
            f"{boxer.name} (力量:{boxer.power} 速度:{boxer.speed} "
            f"體力:{boxer.stamina} 防禦:{boxer.defense} 經驗:{boxer.experience})"
        )

        print(Fore.GREEN + f"{stats:<60}")
        print(f"HP:  {hp_bar} {boxer.current_hp:4.1f}%")
        print(f"ST:  {st_bar} {boxer.current_stamina:4.1f}%")

    # ------------------------------------------------------------
    #  新增: 計算物理優勢（身高、臂展、體重）對命中率與傷害的影響
    # ------------------------------------------------------------
    def calculate_physical_advantages(
        self, attacker: Boxer, defender: Boxer, punch_type: PunchType
    ) -> Tuple[float, float]:
        """
        回傳 (hit_modifier, damage_modifier)
        - hit_modifier: 命中率修正
        - damage_modifier: 傷害修正
        """
        # 臂展差距 => 命中修正
        reach_adv = (attacker.reach - defender.reach) * 0.001 * PUNCH_REACH_IMPORTANCE[
            punch_type
        ]
        # 身高差距 => 對 uppercut 有特殊加成
        height_diff = attacker.height - defender.height
        height_adv = height_diff * PUNCH_HEIGHT_PREFERENCE[punch_type]
        # 體重差距 => 傷害修正
        weight_adv = (
            (attacker.weight - defender.weight)
            * 0.005
            * PUNCH_WEIGHT_IMPORTANCE[punch_type]
        )

        hit_modifier = 1 + reach_adv + height_adv
        damage_modifier = 1 + weight_adv
        return hit_modifier, damage_modifier

    # ------------------------------------------------------------
    #  整合新版 simulate_punch + 防禦/回復邏輯 + 動畫敘述
    # ------------------------------------------------------------
    def simulate_punch(
        self, attacker: Boxer, defender: Boxer, attacker_label: str
    ) -> Dict:
        """
        執行出拳，並回傳動作資訊 (含文字敘述)
        回傳結構示例:
        {
            "is_defense": bool,
            "desc": str,
            "is_hit": bool,
            "is_knockdown": bool,
            "damage": float,
        }
        """
        action_info = {
            "is_defense": False,
            "desc": "",
            "is_hit": False,
            "is_knockdown": False,
            "damage": 0.0,
        }

        # 若體力不足，則防禦並恢復部分體力
        if attacker.current_stamina < 10:
            stamina_recovered = random.uniform(5, 10)
            attacker.current_stamina = min(100, attacker.current_stamina + stamina_recovered)
            action_info["is_defense"] = True
            action_info["desc"] = f"{attacker.name} 防禦並恢復了 {stamina_recovered:.1f} 體力"
            return action_info

        # 隨機選擇拳種
        punch_type = random.choice(list(PunchType))
        punch_info = PUNCH_PROPERTIES[punch_type]

        # 更新回合出拳數
        if attacker_label == "A":
            self.round_state.total_punches_A += 1
            self.round_state.punch_types_A[punch_type.value] += 1
        else:
            self.round_state.total_punches_B += 1
            self.round_state.punch_types_B[punch_type.value] += 1

        # 計算物理優勢
        hit_mod, dmg_mod = self.calculate_physical_advantages(
            attacker, defender, punch_type
        )

        # 命中率計算
        hit_chance = (
            punch_info["accuracy"]
            * (attacker.speed / 100)
            * (attacker.current_stamina / 100)
            * (1 - defender.defense / 200)
            * hit_mod
        )
        is_hit = random.random() < hit_chance
        action_info["is_hit"] = is_hit

        # 若命中，計算傷害
        if is_hit:
            damage = (
                punch_info["damage"]
                * (attacker.power / 100)
                * (attacker.current_stamina / 100)
                * (1 - defender.chin / 220)
                * 0.80  # 額外降傷係數
                * dmg_mod
            )
            defender.current_hp = max(0, defender.current_hp - damage)
            action_info["damage"] = damage

            # 更新回合統計
            if attacker_label == "A":
                self.round_state.clean_hits_A += 1
                self.round_state.points_A += punch_info["points"]
                self.round_state.damage_dealt_A += damage
            else:
                self.round_state.clean_hits_B += 1
                self.round_state.points_B += punch_info["points"]
                self.round_state.damage_dealt_B += damage

            # 計算擊倒機率
            knockout_chance = (
                0.005
                * (damage / 18)
                * (1 - defender.current_hp / 100)
                * (attacker.power / 100)
                * (1 - defender.chin / 140)
                * (1 - defender.current_stamina / 180)
                * dmg_mod
            )
            if random.random() < knockout_chance:
                action_info["is_knockdown"] = True
                defender.current_hp = max(0, defender.current_hp - 12)
                if attacker_label == "A":
                    self.round_state.knockdowns_B += 1
                else:
                    self.round_state.knockdowns_A += 1

            if action_info["is_knockdown"]:
                action_info[
                    "desc"
                ] = f"{attacker.name} 使用 {punch_type.value}，造成 KNOCKDOWN!"
            else:
                action_info[
                    "desc"
                ] = f"{attacker.name} 使用 {punch_type.value}，造成 {damage:.1f} 傷害!"
        else:
            action_info["desc"] = f"{attacker.name} 使用 {punch_type.value}，但未命中!"

        # 攻擊者體力消耗
        stamina_cost = punch_info["stamina_cost"]
        attacker.current_stamina = max(0, attacker.current_stamina - stamina_cost)

        if attacker_label == "A":
            self.round_state.stamina_used_A += stamina_cost
        else:
            self.round_state.stamina_used_B += stamina_cost

        return action_info

    def simulate_round(self) -> Tuple[bool, str, str]:
        """
        模擬一個回合 (3 分鐘 = 180 秒，假設 A->B->A->B... 每輪 2 秒)
        回傳: (match_ended, winner, victory_condition)
        """
        time_left = 180.0
        self.round_state = RoundState()  # 重置本回合數據

        while time_left > 0:
            # 顯示雙方目前狀態
            self.display_stats(time_left)

            # A 出拳
            action_A = self.simulate_punch(self.boxerA, self.boxerB, "A")
            print(Fore.YELLOW + action_A["desc"])
            if (not action_A["is_defense"]) and (self.boxerB.current_hp <= 0):
                # B 被 KO
                time.sleep(2)
                return True, self.boxerA.name, "KO"
            time.sleep(0.5)

            # B 出拳
            action_B = self.simulate_punch(self.boxerB, self.boxerA, "B")
            print(Fore.YELLOW + action_B["desc"])
            if (not action_B["is_defense"]) and (self.boxerA.current_hp <= 0):
                # A 被 KO
                time.sleep(2)
                return True, self.boxerB.name, "KO"
            time.sleep(0.5)

            time_left -= 2

        # 回合結束後處理 (恢復 HP 與 Stamina)
        self._end_round_processing()
        self.round_states.append(self.round_state)
        return False, "", ""

    def _end_round_processing(self):
        """回合結束後的恢復邏輯 (體力 +20, HP +7)"""
        for boxer in [self.boxerA, self.boxerB]:
            boxer.current_stamina = min(100, boxer.current_stamina + 20)
            boxer.current_hp = min(100, boxer.current_hp + 7)

    def _judge_match(self) -> Tuple[str, str]:
        """
        根據分數做最後判決:
          - 分差 <5 => Draw
          - 分差 <40 => Split Decision
          - 否則 => Unanimous Decision
        """
        total_points_A = sum(r.points_A for r in self.round_states)
        total_points_B = sum(r.points_B for r in self.round_states)
        diff = abs(total_points_A - total_points_B)

        if diff < 5:
            return "Draw", "Draw"
        elif diff < 40:
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
        """模擬整場比賽 (可能中途 KO 或打滿回合後判決)"""
        # 進場動畫
        self.clear_screen()
        print(Fore.MAGENTA + f"\n{'=' * self.ring_width}")
        print(Fore.YELLOW + f"拳擊比賽: {self.boxerA.name} vs {self.boxerB.name}")
        print(Fore.MAGENTA + f"{'=' * self.ring_width}")
        time.sleep(2)

        match_ended = False
        winner = ""
        victory_condition = ""

        while self.current_round <= self.total_rounds and not match_ended:
            match_ended, winner, victory_condition = self.simulate_round()
            if not match_ended:
                self.current_round += 1

        # 若沒被KO，則進行判決
        if not match_ended:
            winner, victory_condition = self._judge_match()

        # 顯示最終結果
        self.display_match_result(winner, victory_condition)
        return self._create_match_result(winner, victory_condition)

    def display_match_result(self, winner: str, victory_condition: str):
        """最後清屏並顯示比賽結果"""
        self.clear_screen()
        print(Fore.CYAN + f"\n{'=' * self.ring_width}")
        print(Fore.YELLOW + "比賽結束!")
        print(Fore.CYAN + f"{'=' * self.ring_width}")

        if winner == "Draw":
            print(Fore.YELLOW + f"結果: 比賽平手! ({victory_condition})")
        else:
            print(Fore.YELLOW + f"勝利者: {winner} ({victory_condition})")

        print(Fore.CYAN + f"{'=' * self.ring_width}\n")

    def _create_match_result(self, winner: str, victory_condition: str) -> Dict:
        """
        彙整整場賽事資訊，並補上您需要的欄位：
          - match_id (由外部傳入或自行生成)
          - final_stamina_A, final_stamina_B
          - 各拳種累計 total_A_jab, total_B_jab, ...
        """
        # 累加回合數據
        total_punches_A = sum(r.total_punches_A for r in self.round_states)
        total_punches_B = sum(r.total_punches_B for r in self.round_states)
        total_hits_A = sum(r.clean_hits_A for r in self.round_states)
        total_hits_B = sum(r.clean_hits_B for r in self.round_states)
        total_knockdowns_A = sum(r.knockdowns_A for r in self.round_states)
        total_knockdowns_B = sum(r.knockdowns_B for r in self.round_states)
        total_points_A = sum(r.points_A for r in self.round_states)
        total_points_B = sum(r.points_B for r in self.round_states)
        total_damage_dealt_A = sum(r.damage_dealt_A for r in self.round_states)
        total_damage_dealt_B = sum(r.damage_dealt_B for r in self.round_states)
        total_stamina_used_A = sum(r.stamina_used_A for r in self.round_states)
        total_stamina_used_B = sum(r.stamina_used_B for r in self.round_states)

        # 累計各拳種使用次數
        total_punch_types_A = {p.value: 0 for p in PunchType}
        total_punch_types_B = {p.value: 0 for p in PunchType}
        for r_state in self.round_states:
            for p_type in PunchType:
                total_punch_types_A[p_type.value] += r_state.punch_types_A[p_type.value]
                total_punch_types_B[p_type.value] += r_state.punch_types_B[p_type.value]

        # 若 total_punches_X 為 0，命中率記得避免除以 0
        hit_rate_A = round((total_hits_A / total_punches_A * 100) if total_punches_A > 0 else 0, 2)
        hit_rate_B = round((total_hits_B / total_punches_B * 100) if total_punches_B > 0 else 0, 2)

        # 注意：match_id 可以在 simulate_matches() 或外部注入
        # 這裡預設先放 None，稍後會在 simulate_matches 補上
        result = {
            "match_id": None,
            "winner": winner,
            "victory_condition": victory_condition,
            "rounds_completed": self.current_round,
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
            "total_punches_A": total_punches_A,
            "total_punches_B": total_punches_B,
            "total_hits_A": total_hits_A,
            "total_hits_B": total_hits_B,
            "total_knockdowns_A": total_knockdowns_A,
            "total_knockdowns_B": total_knockdowns_B,
            "total_points_A": total_points_A,
            "total_points_B": total_points_B,
            "total_damage_dealt_A": total_damage_dealt_A,
            "total_damage_dealt_B": total_damage_dealt_B,
            "total_stamina_used_A": total_stamina_used_A,
            "total_stamina_used_B": total_stamina_used_B,
            "hit_rate_A": hit_rate_A,
            "hit_rate_B": hit_rate_B,
            "final_hp_A": round(self.boxerA.current_hp, 2),
            "final_hp_B": round(self.boxerB.current_hp, 2),
            "final_stamina_A": round(self.boxerA.current_stamina, 2),
            "final_stamina_B": round(self.boxerB.current_stamina, 2),
            "total_A_jab": total_punch_types_A["Jab"],
            "total_B_jab": total_punch_types_B["Jab"],
            "total_A_cross": total_punch_types_A["Cross"],
            "total_B_cross": total_punch_types_B["Cross"],
            "total_A_lead_hook": total_punch_types_A["Lead Hook"],
            "total_B_lead_hook": total_punch_types_B["Lead Hook"],
            "total_A_rear_hook": total_punch_types_A["Rear Hook"],
            "total_B_rear_hook": total_punch_types_B["Rear Hook"],
            "total_A_lead_uppercut": total_punch_types_A["Lead Uppercut"],
            "total_B_lead_uppercut": total_punch_types_B["Lead Uppercut"],
            "total_A_rear_uppercut": total_punch_types_A["Rear Uppercut"],
            "total_B_rear_uppercut": total_punch_types_B["Rear Uppercut"],
        }

        return result


# ------------------------------------------------------------
#  範例：多場比賽模擬，可儲存為 CSV
# ------------------------------------------------------------
def simulate_matches(num_matches: int = 5):
    """
    執行多場比賽模擬，每場比賽結束後將結果彙整存到 DataFrame 再輸出為 CSV。
    這裡也會自動產生 match_id (隨機5位數)，並寫入 result。
    """
    results = []
    for i in range(num_matches):
        # 產生 match_id：您可以換成別的邏輯，例如全域遞增或 UUID
        match_id = random.randint(10000, 99999)

        print(f"模擬第 {i+1}/{num_matches} 場比賽 (match_id={match_id})...")

        # 產生兩位拳手
        boxerA = Boxer.generate(f"Boxer_A_{i+1}")
        boxerB = Boxer.generate(f"Boxer_B_{i+1}")

        # 建立比賽
        match = LiveBoxingMatch(boxerA, boxerB, rounds=12)
        match_result = match.simulate_match()

        # 在這裡把 match_id 寫入最終結果
        match_result["match_id"] = match_id

        results.append(match_result)

        if i < num_matches - 1:
            input("\n按下 Enter 進行下一場比賽...")

    # 結束所有比賽後，轉成 DataFrame
    df = pd.DataFrame(results)

    # 建立輸出目錄並存檔
    output_dir = Path("boxing_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"boxing_matches_{num_matches}.csv"
    df.to_csv(output_path, index=False)

    print(f"\n所有比賽數據已保存到: {output_path}")
    print("\n[ 簡易統計 ]")
    print(f"總場次: {num_matches}")
    print(f"KO率: {(df['victory_condition'] == 'KO').mean() * 100:.2f}%")
    print(f"判決勝率: {(df['victory_condition'].str.contains('Decision')).mean() * 100:.2f}%")
    print(f"平均回合數: {df['rounds_completed'].mean():.2f}")
    print(f"平均命中率A: {df['hit_rate_A'].mean():.2f}%")
    print(f"平均命中率B: {df['hit_rate_B'].mean():.2f}%")


# ------------------------------------------------------------
#  主程式測試
# ------------------------------------------------------------
if __name__ == "__main__":
    # (A) 範例：一次模擬 3 場比賽 (可自行調整)
    #simulate_matches(3)

    # (B) 若想要只模擬單場，或客製名人對決，也可以直接建立兩位拳手：
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
        experience=95,
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
        experience=98,
    )
    
    match = LiveBoxingMatch(tyson, ali, rounds=1)
    result = match.simulate_match()
    result["match_id"] = 99999  # 也可以自行指定
    print(result)
