import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, WARPGATE, CHRONOBOOST, AbilityId
from sc2.position import Point2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

class MyBot(sc2.BotAI):
    async def on_step(self, iteration):
        if iteration == 0:
            await self.chat_send("(glhf)")

        await self.distribute_workers()
        await self.build_workers()
        await self.build_supply()
        await self.build_structures()
        await self.train_units()
        await self.warp_units()
        await self.attack()

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    async def build_supply(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_structures(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.can_afford(GATEWAY) and not self.units(GATEWAY).exists:
                await self.build(GATEWAY, near=pylon)

            if self.can_afford(ASSIMILATOR) and self.units(GATEWAY).ready.exists:
                vespene_geyser = self.state.vespene_geyser.closest_to(pylon)
                worker = self.select_build_worker(vespene_geyser.position)
                if worker is not None:
                    await self.do(worker.build(ASSIMILATOR, vespene_geyser))

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(RESEARCH_WARPGATE) and not self.already_pending(RESEARCH_WARPGATE):
                    cybercore = self.units(CYBERNETICSCORE).ready.first
                    await self.do(cybercore(RESEARCH_WARPGATE))

    async def train_units(self):
        for gateway in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(STALKER):
                await self.do(gateway.train(STALKER))

    async def warp_units(self):
        for warpgate in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            if AbilityId.WARPGATETRAIN_STALKER in abilities:
                pos = warpgate.position.to2.random_on_distance(4)
                await self.do(warpgate.warp_in(STALKER, pos))

    async def attack(self):
        if self.units(STALKER).amount > 15:
            target = self.known_enemy_units.closest_to(self.enemy_start_locations[0])
            for stalker in self.units(STALKER).idle:
                await self.do(stalker.attack(target))

def main():
    sc2.run_game(sc2.maps.get("(2)CatalystLE"), [
        Bot(Race.Protoss, MyBot()),
        Computer(Race.Protoss, Difficulty.Easy)
    ], realtime=False)

if __name__ == '__main__':
    main()