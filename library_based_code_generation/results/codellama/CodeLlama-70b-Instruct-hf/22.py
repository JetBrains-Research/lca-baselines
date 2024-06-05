

```
from sc2 import BotAI, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit

class ZergRushBot(BotAI):
    def __init__(self):
        super().__init__()
        self.actions = []
        self.vespene_geyser_tags = set()

    async def on_start(self):
        self.client.game_step = 2

    async def on_step(self, iteration):
        self.actions = []

        await self.chat_message()
        await self.attack_enemy_structures()
        await self.inject_hatcheries()
        await self.manage_resources()
        await self.research_upgrades()
        await self.train_units()
        await self.build_structures()

        await self.do_actions(self.actions)

    async def chat_message(self):
        await self.chat_send("(glhf)")

    async def attack_enemy_structures(self):
        enemy_structures = self.enemy_structures
        if enemy_structures:
            for unit in self.units:
                self.actions.append(unit.attack(enemy_structures.closest_to(unit)))

    async def inject_hatcheries(self):
        for hatchery in self.structures(UnitTypeId.HATCHERY):
            if hatchery.is_ready:
                larvae = self.larva
                if larvae:
                    self.actions.append(larvae.random.inject_larva(hatchery))

    async def manage_resources(self):
        await self.manage_vespene_gas()
        await self.manage_minerals()

    async def manage_vespene_gas(self):
        for vespene_geyser in self.vespene_geyser:
            if not self.vespene_geyser_tags:
                self.vespene_geyser_tags.add(vespene_geyser.tag)

            if self.can_afford(UnitTypeId.EXTRACTOR):
                worker = self.select_build_worker(vespene_geyser.position)
                if worker:
                    self.actions.append(worker.build(UnitTypeId.EXTRACTOR, vespene_geyser))

    async def manage_minerals(self):
        for mineral_field in self.mineral_field:
            if self.can_afford(UnitTypeId.DRONE):
                worker = self.select_build_worker(mineral_field.position)
                if worker:
                    self.actions.append(worker.gather(mineral_field))

    async def research_upgrades(self):
        if self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
            self.research(UpgradeId.ZERGLINGMOVEMENTSPEED)

    async def train_units(self):
        if self.can_afford(UnitTypeId.ZERGLING):
            larvae = self.larva
            if larvae:
                self.actions.append(larvae.random.train(UnitTypeId.ZERGLING))

    async def build_structures(self):
        if self.can_afford(UnitTypeId.HATCHERY):
            for expansion_location in self.expansion_locations_list:
                if not self.structures(UnitTypeId.HATCHERY).closer_than(10, expansion_location):
                    worker = self.select_build_worker(expansion_location)
                    if worker:
                        self.actions.append(worker.build(UnitTypeId.HATCHERY, expansion_location))

    async def on_end(self, game_result):
        print(f"Game ended with result: {game_result}")

    def draw_creep_pixelmap(self):
        for (x, y) in self.state.creep.tumors:
            self._client.debug_box2_out(Point2((x, y)), Point2((x + 1, y + 1)))

run_game(
    maps.get("Abyssal Reef LE"),
    [Bot(Race.Zerg, ZergRushBot()),
     Computer(Race.Terran, Difficulty.Medium)],
    realtime=False,
    save_replay_as="ZergRushBot.SC2Replay"
)
```

This code creates a StarCraft II bot using the python-sc2 library. The bot is a Zerg race bot that performs a rush strategy. The bot has methods to handle the start of the game, each step of the game, and the end of the game.

On start, the bot sets the game step to 2. On each step, the bot performs a series of actions such as sending a chat message, attacking enemy structures, injecting hatcheries with larva, managing vespene gas and mineral resources, researching upgrades, training units, and building structures.

The bot also has a method to draw a creep pixelmap for debugging purposes. At the end of the game, the bot logs that the game has ended. Finally, the bot is run on a specific map against a computer opponent of the Terran race with medium difficulty. The game is not run in real time, and a replay of the game is saved.