import random
import time
from typing import List, Tuple
from dataclasses import dataclass, field
from collections import deque
import asyncio
from datetime import datetime
import csv
import os

class WeightClass:
    FLYWEIGHT = "羽量級"
    LIGHTWEIGHT = "輕量級"
    WELTERWEIGHT = "沉量級"
    MIDDLEWEIGHT = "中量級"
    HEAVYWEIGHT = "重量級"
    
    @classmethod
    def get_weight_class(cls, weight: float) -> str:
        if weight < 60:
            return cls.FLYWEIGHT
        elif weight < 70:
            return cls.LIGHTWEIGHT
        elif weight < 80:
            return cls.WELTERWEIGHT
        elif weight < 90:
            return cls.MIDDLEWEIGHT
        else:
            return cls.HEAVYWEIGHT

@dataclass
class FighterStats:
    name: str
    weight: float
    height: float
    reach: float
    age: int
    wins: int = 0
    losses: int = 0
    knockouts: int = 0
    
    @property
    def weight_class(self) -> str:
        return WeightClass.get_weight_class(self.weight)
    
    @property
    def reach_advantage(self) -> float:
        return self.reach / self.height
    
    @property
    def power_factor(self) -> float:
        # 以體重70kg、身高175cm當作基準
        return (self.weight / 70) * (self.height / 175)
        
    @property
    def experience_factor(self) -> float:
        total_fights = self.wins + self.losses
        if total_fights == 0:
            return 1.0
        win_rate = self.wins / total_fights
        ko_rate = self.knockouts / total_fights if total_fights > 0 else 0
        return (1 + win_rate + ko_rate) / 3

@dataclass
class Fighter:
    stats: FighterStats
    hp: int = 100
    stamina: int = 100
    recent_damage: deque = field(default_factory=lambda: deque(maxlen=5))
    combo_count: int = 0
    is_blocking: bool = False
    block_skill: float = 0.6
    counter_skill: float = 0.3
    
    def reset_combo(self):
        self.combo_count = 0
        
    def update_combo(self):
        self.combo_count += 1
    
    def is_alive(self) -> bool:
        return self.hp > 0

    def attempt_block(self, attack_type: str) -> Tuple[bool, bool]:
        """嘗試格擋攻擊，返回(是否格擋成功, 是否觸發反擊)"""
        if not self.is_blocking:
            return False, False
            
        block_chance = self.block_skill * (self.stamina / 100)
        if attack_type == "重拳":
            block_chance *= 0.7  # 重拳較難格擋
        
        blocked = (random.random() < block_chance)
        counter = False
        if blocked:
            # 有機會反擊
            counter = (random.random() < self.counter_skill)
        # 格擋消耗體力
        self.stamina = max(0, self.stamina - 2)
        return blocked, counter

    def take_damage(self, damage: int, is_counter: bool = False):
        """承受傷害，不考慮 TKO，純粹扣血"""
        if is_counter:
            damage = int(damage * 1.25)  # 反擊傷害加成
        
        # 根據體力值和體重級別微調所受傷害
        stamina_factor = max(0.8, self.stamina / 100)
        weight_factor = 1 + (0.1 * (90 - self.stats.weight) / 90)  
        # 體重越輕，受到的傷害越高 (例如 <90kg的時候)
        
        adjusted_damage = int(damage * weight_factor / stamina_factor)
        
        self.hp = max(0, self.hp - adjusted_damage)
        self.stamina = max(0, self.stamina - 5)
        self.recent_damage.append(adjusted_damage)

    def calculate_damage_multiplier(self, defender: 'Fighter') -> float:
        """計算攻擊傷害倍率"""
        reach_diff = (self.stats.reach - defender.stats.reach) / 100
        reach_multiplier = 1 + (0.1 * reach_diff)
        weight_multiplier = self.stats.power_factor
        exp_multiplier = self.stats.experience_factor
        return reach_multiplier * weight_multiplier * exp_multiplier
    
    def rest(self):
        """回合結束時的休息機制，可視需要決定是否使用"""
        self.stamina = min(100, self.stamina + 10)
        self.is_blocking = False

class BoxingRound:
    def __init__(self, fighter1: Fighter, fighter2: Fighter, match_id: int, logger: 'MatchLogger'):
        self.fighter1 = fighter1
        self.fighter2 = fighter2
        
        # 只有一回合 => 預設60秒 (可改成180)
        self.time_left = 60
        self.start_time = time.time()
        
        # 提高攻擊基底傷害
        self.actions = {
            "直拳":  (12, 22),
            "勾拳": (20, 30),
            "重拳": (30, 45),
            "防禦": None,
            "休息": None
        }

        # 回合是否結束的旗標
        self.is_finished = False
        
        # 新增 logger 與 match_id
        self.logger = logger
        self.match_id = match_id

    def should_take_action(self, fighter: Fighter) -> bool:
        """簡單決定是否要行動的機率"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 時間越久，出招機率稍微提高，最多0.9
        action_probability = min(0.9, 0.2 + elapsed / 20)
        
        if fighter.stamina < 30:
            action_probability *= 0.6
        elif fighter.stamina < 50:
            action_probability *= 0.8
        
        return random.random() < action_probability
    
    def choose_action(self, fighter: Fighter) -> str:
        """根據血量、體力來選擇動作"""
        weights = {
            "直拳": 4,
            "勾拳": 3,
            "重拳": 2,
            "防禦": 1,
            "休息": 1
        }
        
        if fighter.stamina < 30:
            weights["休息"] *= 6
            weights["防禦"] *= 2
            weights["重拳"] *= 0.2
        elif fighter.stamina < 50:
            weights["休息"] *= 3
            weights["重拳"] *= 0.5
            
        if fighter.hp < 40:
            # 血少了，可能更傾向防禦或休息
            weights["防禦"] *= 3
            weights["休息"] *= 2
            weights["直拳"] *= 0.5
            weights["勾拳"] *= 0.3
            weights["重拳"] *= 0.2
        
        actions = list(weights.keys())
        action_weights = list(weights.values())
        return random.choices(actions, weights=action_weights, k=1)[0]

    def print_fighters_status(self):
        """印出雙方最新的 HP 與 Stamina"""
        print(
            f"\n[{self.fighter1.stats.name}] HP: {self.fighter1.hp}, SP: {self.fighter1.stamina}"
            f"   ||   [{self.fighter2.stats.name}] HP: {self.fighter2.hp}, SP: {self.fighter2.stamina}\n"
        )

    def check_knockout(self, attacker: Fighter, defender: Fighter):
        """檢查是否有人被擊倒 (HP <= 0)，若有則結束回合"""
        if not defender.is_alive():  # defender hp <= 0
            print(f"\n!!! {defender.stats.name} 已被擊倒，{attacker.stats.name} 獲勝 !!!")
            self.is_finished = True

        elif not attacker.is_alive():  # 也可能是反擊後 attacker hp <= 0
            print(f"\n!!! {attacker.stats.name} 已被擊倒，{defender.stats.name} 獲勝 !!!")
            self.is_finished = True

    async def handle_action(self, attacker: Fighter, defender: Fighter, action: str):
        """執行一次動作，並立即印出結果，同時記錄到 Logger"""
        # 模擬出招時間
        await asyncio.sleep(0.3)
        
        # 先記錄攻防雙方「動作前」的狀態
        attacker_hp_before = attacker.hp
        defender_hp_before = defender.hp
        attacker_sp_before = attacker.stamina
        defender_sp_before = defender.stamina
        attacker_combo_before = attacker.combo_count
        defender_combo_before = defender.combo_count

        # 執行動作
        if action == "防禦":
            attacker.is_blocking = True
            attacker.reset_combo()
            print(f"{attacker.stats.name} 採用 [防禦] 姿態")

            # 本次動作造成的最終傷害=0
            base_damage = 0
            final_damage = 0
            is_blocked = False
            is_counter = False
            
            self.print_fighters_status()

        elif action == "休息":
            attacker.stamina = min(100, attacker.stamina + 10)
            attacker.is_blocking = False
            attacker.reset_combo()
            print(f"{attacker.stats.name} 選擇 [休息]，恢復體力 ({attacker.stamina}/100)")

            # 本次動作造成的最終傷害=0
            base_damage = 0
            final_damage = 0
            is_blocked = False
            is_counter = False
            
            self.print_fighters_status()
        else:
            # 攻擊動作
            print(f"{attacker.stats.name} 使出 [{action}]！")
            attacker.is_blocking = False

            # 判斷對方是否格擋或反擊
            blocked, counter = defender.attempt_block(action)
            base_damage = random.randint(*self.actions[action]) if self.actions[action] else 0
            is_blocked = blocked
            is_counter = counter

            if blocked:
                print(f"→ {defender.stats.name} 成功格擋了攻擊！")
                attacker.reset_combo()
                
                final_damage = 0  # 若成功格擋，先視為 0
                if counter:
                    print(f"→ {defender.stats.name} 找到 [反擊] 的空檔！")
                    counter_damage = random.randint(10, 20)
                    attacker.take_damage(counter_damage, is_counter=True)
                    defender.update_combo()
                    final_damage = attacker.recent_damage[-1]  # 反擊傷害

                    print(f"→ {defender.stats.name} 的反擊造成 {final_damage} 點傷害！")
                    
                    # 檢查是否因此擊倒了 attacker
                    self.check_knockout(defender, attacker)
                
                self.print_fighters_status()

            else:
                # 沒被格擋 => 正常傷害計算
                damage_multiplier = attacker.calculate_damage_multiplier(defender)
                damage = int(base_damage * damage_multiplier)
                # 若對手 combo_count > 2，也可視為有反擊加成，但這裡簡化處理
                is_counter = (defender.combo_count > 2)

                defender.take_damage(damage, is_counter=is_counter)
                final_damage = defender.recent_damage[-1]

                # combo 邏輯
                attacker.update_combo()
                defender.reset_combo()

                if is_counter:
                    print(f"→ {attacker.stats.name} 攻擊造成 {final_damage} 點傷害！（含反擊加成）")
                else:
                    print(f"→ {attacker.stats.name} 攻擊造成 {final_damage} 點傷害！")

                self.check_knockout(attacker, defender)
                self.print_fighters_status()

        # 完成一次動作後，把「after」的狀態取出
        attacker_hp_after = attacker.hp
        defender_hp_after = defender.hp
        attacker_sp_after = attacker.stamina
        defender_sp_after = defender.stamina
        attacker_combo_after = attacker.combo_count
        defender_combo_after = defender.combo_count

        # 記錄到 Logger 的 event log
        self.logger.log_event(
            match_id=self.match_id,
            action_id=len(self.logger.event_log) + 1,  # 第幾次出招
            timestamp=time.time(),
            attacker_name=attacker.stats.name,
            defender_name=defender.stats.name,
            action=action,
            base_damage=base_damage,
            final_damage=final_damage,
            attacker_hp_before=attacker_hp_before,
            attacker_hp_after=attacker_hp_after,
            defender_hp_before=defender_hp_before,
            defender_hp_after=defender_hp_after,
            attacker_sp_before=attacker_sp_before,
            attacker_sp_after=attacker_sp_after,
            defender_sp_before=defender_sp_before,
            defender_sp_after=defender_sp_after,
            is_blocked=is_blocked,
            is_counter=is_counter,
            attacker_combo_before=attacker_combo_before,
            attacker_combo_after=attacker_combo_after,
            defender_combo_before=defender_combo_before,
            defender_combo_after=defender_combo_after,
        )

    async def run(self):
        """執行唯一一個回合，直到時間耗盡 或 有人被KO"""
        while True:
            elapsed = time.time() - self.start_time
            self.time_left = 60 - int(elapsed)
            
            # 如果有人被擊倒 (is_finished = True)，直接結束
            if self.is_finished:
                break

            # 時間到，也結束回合
            if self.time_left <= 0:
                break
            
            # Fighter1 行動
            if self.should_take_action(self.fighter1):
                action1 = self.choose_action(self.fighter1)
                await self.handle_action(self.fighter1, self.fighter2, action1)
                await asyncio.sleep(0.3)
                
                # 再檢查一次，如果此時已KO就不需要再讓對手出招
                if self.is_finished:
                    break
            
            # Fighter2 行動
            if self.should_take_action(self.fighter2):
                action2 = self.choose_action(self.fighter2)
                await self.handle_action(self.fighter2, self.fighter1, action2)
                await asyncio.sleep(0.3)
            
            # 防止 while 太快跑
            await asyncio.sleep(0.1)

class MatchLogger:
    """
    管理所有比賽的事件記錄 (event log) 與比賽總表 (match summary)
    並在每場比賽結束後，追加寫入 CSV 檔。
    """
    def __init__(self,
                 event_log_file: str = "boxing_events_log.csv",
                 summary_file: str = "boxing_match_summary.csv"):
        self.event_log_file = event_log_file
        self.summary_file = summary_file
        
        # 暫存
        self.event_log = []
        self.match_summaries = []
        
        # 如果檔案不存在，先寫入標題
        if not os.path.exists(self.event_log_file):
            with open(self.event_log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "match_id", "action_id", "timestamp", 
                    "attacker_name", "defender_name", "action",
                    "base_damage", "final_damage",
                    "attacker_hp_before", "attacker_hp_after",
                    "defender_hp_before", "defender_hp_after",
                    "attacker_sp_before", "attacker_sp_after",
                    "defender_sp_before", "defender_sp_after",
                    "is_blocked", "is_counter",
                    "attacker_combo_before", "attacker_combo_after",
                    "defender_combo_before", "defender_combo_after",
                ])
        
        if not os.path.exists(self.summary_file):
            with open(self.summary_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "match_id", "match_date", "winner",
                    "fighter1_name", "fighter1_weight", "fighter1_height", "fighter1_reach", "fighter1_age",
                    "fighter1_power_factor", "fighter1_experience_factor", "fighter1_wins", "fighter1_losses", "fighter1_knockouts",
                    "fighter1_final_hp", "fighter1_final_sp",
                    "fighter2_name", "fighter2_weight", "fighter2_height", "fighter2_reach", "fighter2_age",
                    "fighter2_power_factor", "fighter2_experience_factor", "fighter2_wins", "fighter2_losses", "fighter2_knockouts",
                    "fighter2_final_hp", "fighter2_final_sp",
                    "ko_happened",
                ])
    
    def log_event(self, **kwargs):
        """把一次動作 (attack/block/rest) 的細節記到 event_log"""
        self.event_log.append(kwargs)
    
    def save_events_to_csv(self):
        """將 event_log 追加寫入 CSV"""
        if not self.event_log:
            return
        with open(self.event_log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for event in self.event_log:
                writer.writerow([
                    event.get("match_id"),
                    event.get("action_id"),
                    event.get("timestamp"),
                    event.get("attacker_name"),
                    event.get("defender_name"),
                    event.get("action"),
                    event.get("base_damage"),
                    event.get("final_damage"),
                    event.get("attacker_hp_before"),
                    event.get("attacker_hp_after"),
                    event.get("defender_hp_before"),
                    event.get("defender_hp_after"),
                    event.get("attacker_sp_before"),
                    event.get("attacker_sp_after"),
                    event.get("defender_sp_before"),
                    event.get("defender_sp_after"),
                    event.get("is_blocked"),
                    event.get("is_counter"),
                    event.get("attacker_combo_before"),
                    event.get("attacker_combo_after"),
                    event.get("defender_combo_before"),
                    event.get("defender_combo_after"),
                ])
        # 寫完就清空暫存
        self.event_log = []
    
    def log_match_summary(self, **kwargs):
        """一場比賽結束後，記錄最終結果和選手屬性等"""
        self.match_summaries.append(kwargs)
    
    def save_summaries_to_csv(self):
        """將 match_summaries 追加寫入 CSV"""
        if not self.match_summaries:
            return
        with open(self.summary_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for summary in self.match_summaries:
                writer.writerow([
                    summary.get("match_id"),
                    summary.get("match_date"),
                    summary.get("winner"),
                    summary.get("fighter1_name"),
                    summary.get("fighter1_weight"),
                    summary.get("fighter1_height"),
                    summary.get("fighter1_reach"),
                    summary.get("fighter1_age"),
                    summary.get("fighter1_power_factor"),
                    summary.get("fighter1_experience_factor"),
                    summary.get("fighter1_wins"),
                    summary.get("fighter1_losses"),
                    summary.get("fighter1_knockouts"),
                    summary.get("fighter1_final_hp"),
                    summary.get("fighter1_final_sp"),
                    summary.get("fighter2_name"),
                    summary.get("fighter2_weight"),
                    summary.get("fighter2_height"),
                    summary.get("fighter2_reach"),
                    summary.get("fighter2_age"),
                    summary.get("fighter2_power_factor"),
                    summary.get("fighter2_experience_factor"),
                    summary.get("fighter2_wins"),
                    summary.get("fighter2_losses"),
                    summary.get("fighter2_knockouts"),
                    summary.get("fighter2_final_hp"),
                    summary.get("fighter2_final_sp"),
                    summary.get("ko_happened"),
                ])
        # 寫完就清空暫存
        self.match_summaries = []

class OneRoundBoxingGame:
    def __init__(self, fighter1: FighterStats, fighter2: FighterStats, match_id: int, logger: MatchLogger):
        self.fighter1 = Fighter(fighter1)
        self.fighter2 = Fighter(fighter2)
        self.match_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.match_id = match_id
        self.logger = logger

    async def start_game(self):
        print(f"比賽開始：{self.fighter1.stats.name} ({self.fighter1.stats.weight_class}) vs {self.fighter2.stats.name} ({self.fighter2.stats.weight_class})")
        print(f"時間：60秒，攻擊傷害已提高，看看誰能在一回合內拼出優勢！\n")
        await asyncio.sleep(1)
        
        round_game = BoxingRound(self.fighter1, self.fighter2, self.match_id, self.logger)
        await round_game.run()
        
        # 回合結束後，若沒出現 KO，再判斷血量
        ko_happened = round_game.is_finished  # 只要有人 HP <= 0，就算 KO
        winner = None
        if not round_game.is_finished:
            print(f"\n==== 一回合結束！ ====")
            print(f"{self.fighter1.stats.name}：HP = {self.fighter1.hp}, SP = {self.fighter1.stamina}")
            print(f"{self.fighter2.stats.name}：HP = {self.fighter2.hp}, SP = {self.fighter2.stamina}")
            
            if self.fighter1.hp > self.fighter2.hp:
                print(f">>> {self.fighter1.stats.name} 獲勝！")
                winner = self.fighter1.stats.name
            elif self.fighter2.hp > self.fighter1.hp:
                print(f">>> {self.fighter2.stats.name} 獲勝！")
                winner = self.fighter2.stats.name
            else:
                print(f">>> 雙方平手！")
                winner = "Draw"
        else:
            # 在 check_knockout() 裡我們會印出誰被擊倒
            # 這裡再判斷誰還活著就是贏家
            if self.fighter1.is_alive() and not self.fighter2.is_alive():
                winner = self.fighter1.stats.name
            elif self.fighter2.is_alive() and not self.fighter1.is_alive():
                winner = self.fighter2.stats.name
            else:
                winner = "Draw"
        
        if winner is None:
            winner = "Draw"

        # 紀錄本場比賽的 summary
        self.logger.log_match_summary(
            match_id=self.match_id,
            match_date=self.match_date,
            winner=winner,
            fighter1_name=self.fighter1.stats.name,
            fighter1_weight=self.fighter1.stats.weight,
            fighter1_height=self.fighter1.stats.height,
            fighter1_reach=self.fighter1.stats.reach,
            fighter1_age=self.fighter1.stats.age,
            fighter1_power_factor=self.fighter1.stats.power_factor,
            fighter1_experience_factor=self.fighter1.stats.experience_factor,
            fighter1_wins=self.fighter1.stats.wins,
            fighter1_losses=self.fighter1.stats.losses,
            fighter1_knockouts=self.fighter1.stats.knockouts,
            fighter1_final_hp=self.fighter1.hp,
            fighter1_final_sp=self.fighter1.stamina,
            fighter2_name=self.fighter2.stats.name,
            fighter2_weight=self.fighter2.stats.weight,
            fighter2_height=self.fighter2.stats.height,
            fighter2_reach=self.fighter2.stats.reach,
            fighter2_age=self.fighter2.stats.age,
            fighter2_power_factor=self.fighter2.stats.power_factor,
            fighter2_experience_factor=self.fighter2.stats.experience_factor,
            fighter2_wins=self.fighter2.stats.wins,
            fighter2_losses=self.fighter2.stats.losses,
            fighter2_knockouts=self.fighter2.stats.knockouts,
            fighter2_final_hp=self.fighter2.hp,
            fighter2_final_sp=self.fighter2.stamina,
            ko_happened=ko_happened
        )

        # 寫入 CSV
        self.logger.save_events_to_csv()
        self.logger.save_summaries_to_csv()


# === 這裡是重點：定義 10 位選手，並在 main() 裡隨機抽兩位對打 ===
async def main():
    # 建立 10 位選手資料
    fighters_stats_list = [
        FighterStats(name="張三",   weight=75, height=175, reach=180, age=28, wins=5,  losses=2,  knockouts=3),
        FighterStats(name="李四",   weight=82, height=180, reach=185, age=25, wins=7,  losses=3,  knockouts=4),
        FighterStats(name="王五",   weight=68, height=172, reach=176, age=27, wins=10, losses=4,  knockouts=6),
        FighterStats(name="陳六",   weight=90, height=185, reach=190, age=29, wins=8,  losses=2,  knockouts=5),
        FighterStats(name="林七",   weight=60, height=170, reach=175, age=24, wins=4,  losses=1,  knockouts=2),
        FighterStats(name="趙八",   weight=78, height=177, reach=183, age=31, wins=12, losses=5,  knockouts=7),
        FighterStats(name="孫九",   weight=86, height=182, reach=188, age=26, wins=9,  losses=3,  knockouts=5),
        FighterStats(name="周十",   weight=69, height=171, reach=172, age=23, wins=3,  losses=2,  knockouts=1),
        FighterStats(name="吳十一", weight=74, height=176, reach=179, age=30, wins=6,  losses=5,  knockouts=3),
        FighterStats(name="鄭十二", weight=83, height=178, reach=182, age=32, wins=15, losses=10, knockouts=7),
    ]
    
    # 建立 logger
    logger = MatchLogger(
        event_log_file="boxing_events_log.csv",
        summary_file="boxing_match_summary.csv"
    )
    
    # 取得使用者輸入的比賽場數
    num_matches = int(input("請輸入要進行的比賽場數："))
    
    for match_index in range(1, num_matches + 1):
        print(f"\n====== 第 {match_index} 場比賽 ======\n")
        
        # 從選手池中隨機挑兩位，不重複
        fighter1_stats, fighter2_stats = random.sample(fighters_stats_list, 2)
        
        # 建立並開始一場比賽
        game = OneRoundBoxingGame(fighter1_stats, fighter2_stats, match_id=match_index, logger=logger)
        await game.start_game()
        
        print("\n(休息幾秒，準備下一場...)")
        await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(main())
