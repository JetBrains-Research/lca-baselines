 Here is a simplified version of the StarCraft II bot using the python-sc2 library, following your instructions. Please note that this is a basic outline and may need further refinement and optimization. Also, it assumes that you have the necessary dependencies installed.

```python
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.races import Protoss
from sc2.units import UnitTypes
from sc2.builders import BuildOrder
from sc2.constants import ResourceTypes, UnitTypes as UT

class ProtossBot(Bot):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.manage_resources()
        await self.expand()
        await self.build_nexuses()
        await self.build_pylons()
        await self.build_gateways_and_cybernetics_core()
        await self.build_stargates()
        await self.train_units()
        await self.manage_combat()

    async def manage_resources(self):
        for nexus in self.units(Race.Protoss, UT.Nexus):
            if self.can_afford(UnitTypes.Pylon) and self.supply_left < self.supply_cap:
                await self.build(UnitTypes.Pylon)
            if self.can_afford(ResourceTypes.Gas) and self.closest_structure(UT.Nexus).assimilated_minerals > 1500:
                await self.build_gas_near(self.closest_structure(UT.Nexus))

    async def expand(self):
        if len(self.units(Race.Protoss, UT.Nexus)) < 3:
            for expansion_location in maps.get(name="CatalystLE").get_start_locations(self.race):
                if not self.can_afford(UnitTypes.Nexus) or not self.can_afford(ResourceTypes.Gas) or not self.can_afford(ResourceTypes.MineralField):
                    continue
                await self.build(UnitTypes.Nexus, near=expansion_location)

    async def build_nexuses(self):
        for nexus in self.units(Race.Protoss, UT.Nexus):
            if self.can_afford(UnitTypes.Nexus) and self.assimilated_minerals > 1000:
                await self.build(UnitTypes.Nexus)

    async def build_pylons(self):
        if self.can_afford(UnitTypes.Pylon) and self.supply_left < self.supply_cap and len(self.units(Race.Protoss, UT.Pylon)) < 6:
            await self.build(UnitTypes.Pylon)

    async def build_gateways_and_cybernetics_core(self):
        if self.can_afford(UnitTypes.Gateway) and self.can_afford(UnitTypes.CyberneticsCore):
            if len(self.units(Race.Protoss, UT.Gateway)) < 1 or len(self.units(Race.Protoss, UT.CyberneticsCore)) < 1:
                await self.build(UnitTypes.Gateway)
            elif len(self.units(Race.Protoss, UT.Gateway)) >= 1 and len(self.units(Race.Protoss, UT.CyberneticsCore)) >= 1:
                await self.train(UnitTypes.Sentry)

    async def build_stargates(self):
        if self.can_afford(UnitTypes.Stargate) and len(self.units(Race.Protoss, UT.Nexus)) >= 3 and len(self.units(Race.Protoss, UT.Stargate)) < 3:
            await self.build(UnitTypes.Stargate)

    async def train_units(self):
        for nexus in self.units(Race.Protoss, UT.Nexus):
            if self.can_afford(UnitTypes.Probe) and nexus.assimilated_minerals > 500:
                await self.train(UnitTypes.Probe, worker=True, near=nexus)
        for stargate in self.units(Race.Protoss, UT.Stargate):
            if self.can_afford(UnitTypes.VoidRay) and len(self.units(Race.Protoss, UT.Stargate)) >= 3 and len(self.units(Race.Protoss, UT.TownHall)) >= 3:
                await self.train(UnitTypes.VoidRay, near=stargate)

    async def manage_combat(self):
        attackers, defenders = get_attacker_and_defender(self)
        if not attackers and not defenders:
            return
        if not defenders:
            for worker in self.units(Race.Protoss, UT.Probe):
                await worker.attack(attackers[0])
        elif len(self.units(Race.Protoss, UT.Nexus)) > 0:
            for nexus in self.units(Race.Protoss, UT.Nexus):
                nexus.use_ability(nexus.abilities.ChronoBoost, target=nexus)
            for void_ray in self.units(Race.Protoss, UT.VoidRay):
                if len(void_ray.attack(attackers)) > 5:
                    await void_ray.attack(defenders)
        else:
            for void_ray in self.units(Race.Protoss, UT.VoidRay):
                await void_ray.attack(attackers)

if __name__ == "__main__":
    run_game(
        maps.get("CatalystLE"),
        [ProtossBot()],
        Difficulty.Easy,
        realtime=True,
        visualize=False,
        step_mode=True
    )
```

This code creates a Protoss bot that follows the specified strategy. It expands to three bases, builds three stargates, manages resources, trains units, and engages in combat as described. The bot is run on the "(2)CatalystLE" map against an easy difficulty Protoss computer opponent.