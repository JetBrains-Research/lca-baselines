```python
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import HATCHERY, LARVA, SPAWNINGPOOL, ZERGLING
from sc2.bot_ai import BotAI
import logging

class ZergRushBot(BotAI):
    def __init__(self):
        self.game_step = 2

    async def on_start(self):
        self.game_step = 2

    async def on_step(self, iteration):
        if iteration == 0:
            await self.chat_send("Zerg rush kekeke!")

        await self.distribute_workers()
        await self.build_workers()
        await self.build_overlords()
        await self.build_spawning_pool()
        await self.spawn_zerglings()
        await self.attack()

    async def build_workers(self):
        for hatchery in self.structures(HATCHERY).ready.idle:
            if self.can_afford(self.race_worker) and self.supply_left > 0:
                hatchery.train(self.race_worker)

    async def build_overlords(self):
        if self.supply_left < 2 and not self.already_pending(self.supply_unit):
            hatcheries = self.structures(HATCHERY).ready
            if hatcheries.exists and self.can_afford(self.supply_unit):
                hatcheries.random.train(self.supply_unit)

    async def build_spawning_pool(self):
        if not self.structures(SPAWNINGPOOL):
            if self.can_afford(SPAWNINGPOOL) and not self.already_pending(SPAWNINGPOOL):
                await self.build(SPAWNINGPOOL, near=self.townhalls.first)

    async def spawn_zerglings(self):
        if self.structures(SPAWNINGPOOL).ready.exists:
            for larva in self.units(LARVA).idle:
                if self.can_afford(ZERGLING) and self.supply_left > 0:
                    larva.train(ZERGLING)

    async def attack(self):
        if self.units(ZERGLING).amount > 10:
            for zergling in self.units(ZERGLING).idle:
                await self.do(zergling.attack(self.find_target(self.state)))

    def find_target(self, state):
        if len(self.enemy_structures) > 0:
            return self.enemy_structures.random.position
        return self.enemy_start_locations[0]

    async def on_end(self, game_result):
        logging.info("Game ended.")

    async def draw_creep_pixelmap(self):
        # Example function for debugging purposes
        creep_map = self.state.creep
        # Further implementation would go here to visualize the creep map

def main():
    run_game(
        maps.get("AbyssalReefLE"),
        [
            Bot(Race.Zerg, ZergRushBot()),
            Computer(Race.Terran, Difficulty.Medium)
        ],
        realtime=False,
        save_replay_as="ZergRushBot.SC2Replay"
    )

if __name__ == '__main__':
    main()
```