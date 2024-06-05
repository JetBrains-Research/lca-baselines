```python
import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER
from sc2.constants import WARPGATE, WARPGATETRAIN_STALKER
from sc2.position import Point2

class StarCraftIIBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_gateway()
        await self.build_cybercore()
        await self.research_warp_gate()
        await self.warp_new_units()
        await self.attack()

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount < 2 and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_gateway(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).amount < 3 and self.can_afford(GATEWAY):
                await self.build(GATEWAY, near=pylon)

    async def build_cybercore(self):
        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            pylon = self.units(PYLON).ready.random
            if self.can_afford(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def research_warp_gate(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            cybercore = self.units(CYBERNETICSCORE).ready.first
            if not self.already_pending_upgrade(WARPGATE) and self.can_afford(cybercore.research(WARPGATE)):
                await self.do(cybercore.research(WARPGATE))

    async def warp_new_units(self):
        for warpgate in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            if WARPGATETRAIN_STALKER in abilities and self.can_afford(STALKER):
                pos = self.units(PYLON).ready.random.position.to2.random_on_distance(4)
                placement = await self.find_placement(WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    # If no placement point found, skip this iteration
                    continue
                await self.do(warpgate.warp_in(STALKER, placement))

    async def attack(self):
        if self.units(STALKER).amount > 5:
            for stalker in self.units(STALKER).idle:
                await self.do(stalker.attack(self.find_target(self.state)))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return self.known_enemy_units.closest_to(self.units(NEXUS).first).position
        elif len(self.known_enemy_structures) > 0:
            return self.known_enemy_structures.closest_to(self.units(NEXUS).first).position
        else:
            return self.enemy_start_locations[0]

def main():
    run_game(maps.get("(2)CatalystLE"), [
        Bot(Race.Protoss, StarCraftIIBot()),
        Computer(Race.Protoss, Difficulty.Easy)
    ], realtime=False)

if __name__ == '__main__':
    main()
```