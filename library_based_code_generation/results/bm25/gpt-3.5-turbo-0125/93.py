import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STARGATE, VOIDRAY
from sc2.position import Point2

class MyBot(sc2.BotAI):
    async def on_step(self, iteration):
        if iteration == 0:
            await self.chat_send("(glhf)")

        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_structures()
        await self.build_units()
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
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_structures(self):
        if self.units(NEXUS).amount >= 3:
            if self.units(GATEWAY).amount < 1 and self.can_afford(GATEWAY):
                pylon = self.units(PYLON).ready.random
                if pylon is not None:
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).amount < 1 and self.can_afford(CYBERNETICSCORE):
                pylon = self.units(PYLON).ready.random
                if pylon is not None:
                    await self.build(CYBERNETICSCORE, near=pylon)

            if self.units(STARGATE).amount < 3 and self.units(NEXUS).amount >= 3 and self.can_afford(STARGATE):
                pylon = self.units(PYLON).ready.random
                if pylon is not None:
                    await self.build(STARGATE, near=pylon)

    async def build_units(self):
        for gateway in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.units(NEXUS).amount >= 3:
                await self.do(gateway.train(VOIDRAY))

    async def attack(self):
        if self.units(NEXUS).amount == 0:
            for worker in self.workers:
                await self.do(worker.attack(self.enemy_start_locations[0]))
        elif self.units(VOIDRAY).amount > 5:
            for voidray in self.units(VOIDRAY).idle:
                await self.do(voidray.attack(self.enemy_start_locations[0]))

def main():
    sc2.run_game(sc2.maps.get("(2)CatalystLE"), [
        Bot(Race.Protoss, MyBot()),
        Computer(Race.Protoss, Difficulty.Easy)
    ], realtime=False)

if __name__ == '__main__':
    main()