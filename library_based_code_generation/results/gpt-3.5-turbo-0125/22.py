import sc2
from sc2 import Race
from sc2.player import Bot, Computer

class RushZergBot(sc2.BotAI):
    async def on_start(self):
        self.game_step = 2

    async def on_step(self, iteration):
        # Perform actions such as sending chat messages, attacking, injecting hatcheries, managing resources, researching upgrades, training units, building structures
        pass

    def draw_creep_pixelmap(self):
        # Draw creep pixelmap for debugging
        pass

    async def on_end(self, game_result):
        self._client.game_step = 2
        print("Game has ended")

bot = RushZergBot()
sc2.run_game(sc2.maps.get("Abyssal Reef LE"), [
    Bot(Race.Zerg, bot),
    Computer(Race.Terran, Difficulty.Medium)
], realtime=False, save_replay_as="replay.SC2Replay")