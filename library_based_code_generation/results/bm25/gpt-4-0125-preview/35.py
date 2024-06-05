```python
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PYLON, PROBE, GATEWAY, CYBERNETICSCORE, ASSIMILATOR, STALKER
from sc2.position import Point2
import random

class StarCraftIIBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_gas()
        await self.expand()
        await self.build_gateway()
        await self.build_cybernetics_core()
        await self.research_warp_gate()
        await self.build_stalkers()
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

    async def build_gas(self):
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def expand(self):
        if self.units(NEXUS).amount < 2 and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_gateway(self):
        if self.units(PYLON).ready.exists and self.units(GATEWAY).amount < 3:
            pylon = self.units(PYLON).ready.random
            if self.can_afford(GATEWAY):
                await self.build(GATEWAY, near=pylon)

    async def build_cybernetics_core(self):
        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            pylon = self.units(PYLON).ready.random
            if self.can_afford(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def research_warp_gate(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            cybercore = self.units(CYBERNETICSCORE).ready.first
            if not self.already_pending_upgrade(WARPGATERESEARCH) and self.can_afford(WARPGATERESEARCH):
                await self.do(cybercore.research(WARPGATERESEARCH))

    async def build_stalkers(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))

    async def attack(self):
        if self.units(STALKER).amount > 15:
            for s in self.units(STALKER).idle:
                await self.do(s.attack(self.find_target(self.state)))

        elif self.units(STALKER).amount > 3:
            if len(self.known_enemy_units) > 0:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units).position
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures).position
        else:
            return self.enemy_start_locations[0]

run_game(maps.get("(2)CatalystLE"), [
    Bot(Race.Protoss, StarCraftIIBot()),
    Computer(Race.Protoss, Difficulty.Easy)
], realtime=False)
```