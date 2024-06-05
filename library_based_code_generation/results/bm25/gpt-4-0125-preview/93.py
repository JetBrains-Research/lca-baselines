```python
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PYLON, PROBE, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STARGATE, VOIDRAY
from sc2.bot_ai import BotAI

class ProtossBot(BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.expand()
        await self.build_assimilators()
        await self.build_gateway_and_cybernetics()
        await self.build_stargates()
        await self.train_voidrays()
        await self.chrono_boost()
        await self.attack()

    async def build_workers(self):
        for nexus in self.structures(NEXUS).ready.idle:
            if self.can_afford(PROBE) and self.workers.amount < self.townhalls.amount * 22:
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.structures(NEXUS).ready
            if nexuses.exists and self.can_afford(PYLON):
                await self.build(PYLON, near=nexuses.first)

    async def expand(self):
        if self.structures(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_assimilators(self):
        for nexus in self.structures(NEXUS).ready:
            vespenes = self.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.structures(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def build_gateway_and_cybernetics(self):
        if not self.structures(GATEWAY).ready.exists and self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            nexuses = self.structures(NEXUS).ready
            if nexuses.exists:
                await self.build(GATEWAY, near=nexuses.first)
        if self.structures(GATEWAY).ready.exists and not self.structures(CYBERNETICSCORE).exists and self.can_afford(CYBERNETICSCORE):
            await self.build(CYBERNETICSCORE, near=self.structures(GATEWAY).ready.first)

    async def build_stargates(self):
        if self.structures(NEXUS).amount >= 3 and self.structures(STARGATE).amount < 3 and self.can_afford(STARGATE) and not self.already_pending(STARGATE):
            await self.build(STARGATE, near=self.structures(PYLON).closest_to(self.structures(NEXUS).first))

    async def train_voidrays(self):
        for stargate in self.structures(STARGATE).ready.idle:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(stargate.train(VOIDRAY))

    async def chrono_boost(self):
        for nexus in self.structures(NEXUS).ready:
            if not nexus.has_buff:
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    async def attack(self):
        if self.units(VOIDRAY).amount > 5:
            for vr in self.units(VOIDRAY).idle:
                await self.do(vr.attack(self.find_target(self.state)))

        if not self.structures(NEXUS).exists:
            for worker in self.workers:
                await self.do(worker.attack(self.find_target(self.state)))

    def find_target(self, state):
        if self.known_enemy_units:
            return self.known_enemy_units.closest_to(self.start_location)
        elif self.known_enemy_structures:
            return self.known_enemy_structures.closest_to(self.start_location)
        else:
            return self.enemy_start_locations[0]

run_game(maps.get("(2)CatalystLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Protoss, Difficulty.Easy)
], realtime=False)
```