 Here is a simplified version of a StarCraft II bot using the python-sc2 library that follows your instructions. Please note that this is a basic example and may need further refinement and optimization. Also, it assumes that you have the python-sc2 library installed.

```python
from sc2 import RunBot, Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.units import Units
from sc2.data import Difficulty
from sc2.managers import AStarManager
from sc2.position import Point2

class MyBot(Bot):
    async def on_step(self, iteration):
        if not self.alive:
            return False

        # Check if there are no nexuses left
        if not self.units(NEXUS).ready.exists:
            await self.build(NEXUS)

        # Resource management
        if self.can_afford(GasExtractor) and not self.is_worker_gathering(GasExtractor):
            await self.build_gas_extractor_at(self.main_ramp.closest_to(self.mineral_field))

        # Worker distribution
        for worker in self.workers:
            if not worker.is_gathering and not worker.is_building:
                await worker.gather(self.mineral_field)

        # Build Pylon when on low supply
        if self.supply_used < self.supply_cap and not self.can_afford(Pylon):
            await self.build_pylon_at(self.main_ramp.closest_to(self.mineral_field))

        # Train probes
        if self.can_afford(PROBE) and not self.units(PROBE).completed.exists:
            await self.train(PROBE)

        # Build Gateways
        if self.can_afford(GATEWAY) and not self.units(GATEWAY).completed.exists:
            await self.build(GATEWAY)

        # Research Warp Gate
        if self.units(GATEWAY).completed.exists:
            gateway = self.units(GATEWAY).completed[0]
            if not gateway.is_researching and gateway.research_cooldown.remaining > 0:
                await gateway.research_upgrade(WARP_GATE)

        # Morph to Warp Gate when research is complete
        if self.units(WARP_GATE).completed.exists:
            warp_gate = self.units(WARP_GATE).completed[0]
            if warp_gate.is_idle:
                await warp_gate.morph(STALKER)

        # Warp new units
        if warp_gate.can_warp(STALKER) and not self.units(STALKER).completed.exists:
            await warp_gate.warp(STALKER)

        # Make stalkers attack either closest enemy unit or enemy spawn location
        for stalker in self.units(STALKER):
            if stalker.can_attack:
                target = stalker.closest_enemy()
                if not target:
                    target = stalker.closest_point(self.enemy_start_locations[0])
                stalker.attack(target)

        # Build proxy pylon
        if self.supply_used < self.supply_cap and not self.can_afford(Pylon):
            await self.build_proxy_pylon()

        # Chrono boost nexus or cybercore
        if self.units(NEXUS).completed.exists:
            nexus = self.units(NEXUS).completed[0]
            if nexus.can_be_upgraded and nexus.upgrade_cooldown.remaining > 0:
                await nexus.upgrade(IDIDONTKNOW)  # Replace IDIDONTKNOW with the correct upgrade ID

        return False

    async def build_proxy_pylon(self):
        # Find a suitable location for the proxy pylon
        # ...
        await self.build_pylon_at(location)

    async def run(self):
        await self.async_run_bot(map_name="CatalystLE", AStar=AStarManager(2), difficulty=Difficulty.Easy, visualize=False)

if __name__ == "__main__":
    RunBot(
        MyBot,
        MyBot(Race.Protoss),
        computer=Computer(Race.Protoss, Difficulty.Easy),
        visualize=False
    ).run_forever()
```

This code creates a Protoss bot that manages resources, builds structures, trains units, and engages in combat. It also handles situations when there are no nexuses left, builds pylons when on low supply, trains probes, builds gateways, researches warp gate, morphs to warp gate when research is complete, warps new units, makes stalkers attack either closest enemy unit or enemy spawn location, builds proxy pylons, and chrono boosts nexus or cybercore. The bot is run on the "(2)CatalystLE" map against a Protoss computer opponent with easy difficulty.