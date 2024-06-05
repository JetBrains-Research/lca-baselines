 import python_sc2.maps
import random
from python_sc2 import Race, Difficulty, Runner
from python_sc2.data import Race, Upgrade, AbilityId

class ZergRushBot(BotAI):
    async def on_start(self):
        self.step = 2
        await self.chat_send("Zerg Rush Bot has started the game!")

    async def on_step(self, iteration):
        if iteration == self.step:
            await self.do_actions()
            self.step += 1

        if iteration % 10 == 0:
            await self.draw_creep_pixelmap()

    async def do_actions(self):
        await self.chat_send("Performing actions...")
        units = self.units(Race.Zerg.value)
        enemy_structures = self.known_enemy_structures()

        for unit in units:
            if unit.is_idle and unit.can_attack:
                closest_enemy = self.closest_unit_of_race_to_point(Race.Terran, unit.position)
                if closest_enemy and unit.in_attack_range_of(closest_enemy):
                    await self.do_attack(unit, closest_enemy)

            if unit.type_id == Race.Zerg.Larva.value:
                await self.inject_larva(unit)

            if unit.is_idle and unit.mineral_contents > 0:
                await self.train_unit(unit, Race.Zerg.Zergling.value)

            if unit.type_id == Race.Zerg.Hatchery.value or unit.type_id == Race.Zerg.Lair.value or unit.type_id == Race.Zerg.Hive.value:
                await self.extract_vespene(unit)

        for upgrade in [Upgrade.ZergLarvaSpawning, Upgrade.ZergMetabolicBoost]:
            if not self.knows_upgrade(upgrade):
                await self.research_upgrade(upgrade)

        if self.can_afford(Race.Zerg.SpawningPool):
            await self.build_structure(Race.Zerg.SpawningPool, near=self.townhalls.first)

        if self.can_afford(Race.Zerg.Extractor) and self.vespene_geyser.exists:
            await self.build_structure(Race.Zerg.Extractor, near=self.vespene_geyser.first)

    async def draw_creep_pixelmap(self):
        await self.draw_text("Creep Pixelmap", (0, 0))
        for x in range(0, self.game_info.map_size.x):
            for y in range(0, self.game_info.map_size.y):
                if self.creep.pixel_value(x, y):
                    await self.draw_rect((x * 8, y * 8), (8, 8), color=(0, 255, 0))

    async def on_game_end(self, game_result):
        await self.chat_send(f"Game has ended with result: {game_result}")

map_name = "AbyssalReefLE"
bot_race = Race.Zerg
opponent_race = Race.Terran
difficulty = Difficulty.Medium

runner = Runner(
    maps={map_name: python_sc2.maps.get_map(map_name)},
    bot_class=ZergRushBot,
    bot_args={"bot_race": bot_race, "opponent_race": opponent_race},
    custom_game_speed=2,
    save_replay_as="Zerg_Rush_Bot.SC2Replay",
    real_time=False,
    random_seed=random.randint(0, 999999),
)

runner.run(bot_race, opponent_race, difficulty)