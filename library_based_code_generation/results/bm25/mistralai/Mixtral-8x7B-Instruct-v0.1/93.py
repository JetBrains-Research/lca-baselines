 ```python
from sc2 import Bot, Race, Map, Difficulty
from sc2.constants import Nexus, Pylon, Gateway, CyberneticsCore, Assimilator, TwilightCouncil, Stargate, RoboticsFacility, VoidRay, SCV, CommandCenter, Barracks, OrbitalCommand, SupplyProvider, SCV_GATHER_SPEED, TERRAN_COMMAND_CENTER, PROTOSS_NEXUS, PROTOSS_PYLON, PROTOSS_GATEWAY, PROTOSS_CYBERNETICS_CORE, PROTOSS_ASSIMILATOR, PROTOSS_TWILIGHT_COUNCIL, PROTOSS_STARGATE, PROTOSS_ROBOTICS_FACILITY, PROTOSS_VOID_RAY, NEXUS_CHRONO_BOOST_ABLE
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
import random

class ProtossBot(Bot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def on_step(self, iteration):
        if iteration == 0:
            await self.chat_send("Hi, I'm ProtossBot!")

        await self.distribute_workers()
        await self.manage_resources()
        await self.manage_structures()
        await self.train_units()
        await self.attack_if_necessary()

    async def distribute_workers(self):
        for nexus in self.units(Nexus):
            if nexus.is_idle and not nexus.has_mineral_patches:
                await self.select_build_worker(nexus, AbilityId.SMART_RETURN_MINERALS)

            mineral_patches = nexus.mineral_patches.ready.idle
            for mineral_patch in mineral_patches:
                if self.can_afford(SCV) and self.units(SCV).amount < 22:
                    await self.do(mineral_patch.train(SCV))

    async def manage_resources(self):
        if self.units(Nexus).amount < 3 and not self.already_pending(Nexus):
            await self.build_nexus()

        if self.units(Pylon).amount < (self.units(Nexus).amount * 2) + 2:
            await self.build_pylon()

        if self.units(Assimilator).amount < self.units(Nexus).amount:
            await self.build_gas()

        if self.units(Gateway).amount < 2 and not self.already_pending(Gateway):
            await self.build_gateway()

        if self.units(CyberneticsCore).amount < self.units(Nexus).amount:
            await self.build_cybernetics_core()

        if self.units(Stargate).amount < self.units(Nexus).amount - 1 and self.units(Nexus).amount > 2:
            await self.build_stargate()

        if self.units(RoboticsFacility).amount < self.units(Nexus).amount:
            await self.build_robotics_facility()

    async def build_nexus(self):
        await self.build(Nexus, near=self.units(Nexus).random)

    async def build_pylon(self):
        await self.build(Pylon, near=self.units(SupplyProvider).random.position.towards(self.game_info.map_center, 10))

    async def build_gas(self):
        if not self.units(Assimilator).exists:
            nexus = self.units(Nexus).random
            await self.build(Assimilator, near=nexus)
        else:
            for nexus in self.units(Nexus).ready:
                if not nexus.has_mineral_patches and not nexus.has_gas_geyser:
                    geyser = nexus.gas_geyser.random
                    if self.can_afford(Assimilator) and not self.already_pending(Assimilator):
                        await self.build(Assimilator, near=geyser)

    async def build_gateway(self):
        await self.build(Gateway, near=self.units(Nexus).random)

    async def build_cybernetics_core(self):
        await self.build(CyberneticsCore, near=self.units(Nexus).random)

    async def build_stargate(self):
        await self.build(Stargate, near=self.units(Nexus).random)

    async def build_robotics_facility(self):
        await self.build(RoboticsFacility, near=self.units(Nexus).random)

    async def train_units(self):
        if self.units(Stargate).amount >= 3 and self.units(Proto
```