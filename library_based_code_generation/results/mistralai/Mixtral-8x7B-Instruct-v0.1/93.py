 ```
from sc2 import Bot, Race, MapLocation
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
import time

class ProtossBot(Bot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MAP_NAME = "(2)CatalystLE"
        self.RACE = Race.Protoss
        self.DIFFICULTY = 1

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.manage_production()
        await self.manage_army()
        await self.manage_upgrades()
        await self.manage_expansions()

    async def manage_production(self):
        if self.supply_used < self.supply_cap - 5 and self.can_afford(UnitTypeId.PYLON):
            await self.build_unit(UnitTypeId.PYLON, near=self.townhalls.first.position.towards(self.townhalls.random.position, distance=10))
        if len(self.nexuses) < 3 and self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            await self.build_unit(UnitTypeId.NEXUS, near=self.townhalls.random.position.towards(self.townhalls.random.position.towards(self.townhalls.random.position, distance=15), distance=5))
        if len(self.gateways) < 2 and self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
            await self.build_unit(UnitTypeId.GATEWAY, near=self.townhalls.first.position)
        if len(self.gateways) > 0 and len(self.cybernetics_cores) == 0 and self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
            await self.build_unit(UnitTypeId.CYBERNETICSCORE, near=self.townhalls.first.position)
        if len(self.stargates) < 3 and len(self.nexuses) >= 3 and len(self.nexuses) < 4 and self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
            await self.build_unit(UnitTypeId.STARGATE, near=self.townhalls.first.position)
        if len(self.stargates) > 0 and len(self.voidrays) < 5 and self.can_afford(UnitTypeId.VOIDRAY) and not self.already_pending(UnitTypeId.VOIDRAY):
            await self.build_unit(UnitTypeId.VOIDRAY, near=self.stargates.random.position)

    async def manage_army(self):
        if self.units(UnitTypeId.NEXUS).ready.exists and self.units(UnitTypeId.PROBE).amount < 22 and not self.already_pending(UnitTypeId.PROBE):
            await self.train(UnitTypeId.PROBE, self.units(UnitTypeId.NEXUS).ready.first)
        if self.units(UnitTypeId.STARGATE).ready.exists and self.units(UnitTypeId.VOIDRAY).amount < 20 and not self.already_pending(UnitTypeId.VOIDRAY):
            await self.train(UnitTypeId.VOIDRAY, self.units(UnitTypeId.STARGATE).ready.first)
        if not self.units(UnitTypeId.ARMY).exists and self.units(UnitTypeId.PROBE).amount > 15:
            for probe in self.units(UnitTypeId.PROBE).idle:
                await probe.attack(self.enemy_start_locations[0])
        if self.units(UnitTypeId.VOIDRAY).amount > 5 and self.units(UnitTypeId.ARMY).amount > 0:
            for voidray in self.units(UnitTypeId.VOIDRAY).idle:
                await voidray.attack(self.enemy_units.first)

    async def manage_upgrades(self):
        if self.can_afford(UpgradeId.WARPGATE) and not self.research_in_progress(UpgradeId.WARPGATE):
            await self.research(UpgradeId.WARPGATE)

    async def manage_expansions(self):
        if len(self.nexuses) < 3 and self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            await self.expand_now()

    async def on_unit_completed(self, unit: Unit):
        if unit.type_id == UnitTypeId.NEXUS:
            if self.units(UnitTypeId.PYLON).amount < self.units(UnitTypeId.NEXUS).amount + 2:
                await self.build_unit(UnitTypeId.PYLON, near=unit.position)
            for nexus in self.units(UnitTypeId.NEXUS).ready:
                if not nexus.has_buff(BuffId.CHRONOBOOST):
                    await self.do(nexus(AbilityId.CHRONOBOOST_NEXUS))
        if unit.type_id == UnitTypeId.GATEWAY:
            if self.units(UnitTypeId.CYBERNETICSCORE).amount > 0:
                await self.train(UnitTypeId.CYBORG, unit)
        if unit.type_id == UnitTypeId.CYBERNETICSCORE:
            if self.units(UnitTypeId.GATEWAY).amount > 1:
                await self.train(UnitTypeId.WARPGATE, unit)
        if unit.type_id == UnitTypeId.STARGATE:
            if self.units(UnitTypeId.VOIDRAY).amount < 5:
                await self.train(UnitTypeId.VOIDRAY, unit)

if __name__ == "__main__":
    bot = ProtossBot()
    bot.start(maps=[MapLocation(bot.MAP_NAME)], difficulty=bot.DIFFICULTY)
```
Please note that this is a basic implementation and may not be optimal. The bot may not perform well against higher difficulty AI opponents. Also, the bot does not handle all possible edge cases and may behave unexpectedly in certain situations.