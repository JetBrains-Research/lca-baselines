```python
import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PYLON, PROBE, GATEWAY, CYBERNETICSCORE, ASSIMILATOR, STARGATE, VOIDRAY
from sc2.position import Point2

class ProtossBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.expand()
        await self.build_gas()
        await self.build_gateway_and_cybernetics()
        await self.build_stargate()
        await self.chrono_boost()
        await self.train_voidrays()
        await self.attack()

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE) and nexus.assigned_harvesters < nexus.ideal_harvesters:
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                await self.build(PYLON, near=nexuses.first)

    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_gas(self):
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

    async def build_gateway_and_cybernetics(self):
        if not self.units(GATEWAY).ready.exists and not self.already_pending(GATEWAY):
            if self.can_afford(GATEWAY) and self.units(PYLON).ready.exists:
                await self.build(GATEWAY, near=self.units(PYLON).ready.random)

        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and self.units(PYLON).ready.exists:
                await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.random)

    async def build_stargate(self):
        if self.units(NEXUS).amount >= 3 and self.units(STARGATE).amount < 3:
            if self.can_afford(STARGATE) and self.units(PYLON).ready.exists:
                await self.build(STARGATE, near=self.units(PYLON).ready.random)

    async def chrono_boost(self):
        for nexus in self.units(NEXUS).ready:
            if not nexus.has_buff:
                abilities = await self.get_available_abilities(nexus)
                if sc2.constants.EffectId.CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(sc2.constants.EffectId.CHRONOBOOSTENERGYCOST, nexus))

    async def train_voidrays(self):
        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.units(NEXUS).amount >= 3:
                await self.do(stargate.train(VOIDRAY))

    async def attack(self):
        if self.units(VOIDRAY).amount > 5:
            for vr in self.units(VOIDRAY).idle:
                await self.do(vr.attack(self.find_target(self.state)))

        if self.units(NEXUS).amount == 0:
            for worker in self.units(PROBE):
                await self.do(worker.attack(self.find_target(self.state)))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return self.known_enemy_units.closest_to(self.units(NEXUS).first)
        elif len(self.known_enemy_structures) > 0:
            return self.known_enemy_structures.closest_to(self.units(NEXUS).first)
        else:
            return self.enemy_start_locations[0]

run_game(maps.get("(2)CatalystLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Protoss, Difficulty.Easy)
], realtime=False)
```