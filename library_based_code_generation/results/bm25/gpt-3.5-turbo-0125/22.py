import sc2
from sc2 import Race
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.data import race_townhalls

class RushZergBot(sc2.BotAI):
    async def on_start(self):
        self.game_step = 2

    async def on_step(self, iteration):
        # Perform rush strategy actions
        pass

    async def draw_creep_pixelmap(self):
        # Draw creep pixelmap for debugging
        pass

    async def on_end(self, game_result):
        self._client.game_step = 8
        print("Game ended")

bot = Bot(Race.Zerg, RushZergBot())
computer_opponent = Computer(Race.Terran, Difficulty.Medium)
sc2.run_game(sc2.maps.get("Abyssal Reef LE"), [bot, computer_opponent], realtime=False, save_replay_as="replay.SC2Replay")