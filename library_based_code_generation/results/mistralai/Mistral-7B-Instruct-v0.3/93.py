 Here is a basic outline of the StarCraft II bot using the python-sc2 library. This code is a simplified version and might need adjustments to work properly.

```python
from sc2 import RunBot, Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.units import Units
from sc2.data import Race, UnitTypeId
from sc2.managers import ResourceManager

class ProtossBot(RunBot):
    async def on_step(self, iteration):
        self.distribute_workers()
        self.manage_resources()
        self.expand()
        self.build_structures()
        self.train_units()
        self.attack()

    def distribute_workers(self):
        for nexus in self.units(Nexus):
            if nexus.assigned_build_queue:
                continue
            for worker in self.units(Probe):
                if worker.idle and not worker.is_gathering:
                    worker.gather(nexus)

    def manage_resources(self):
        if self.can_afford(Pylon) and self.supply_left < self.supply_cap:
            self.build(Pylon)

        for nexus in self.units(Nexus):
            if nexus.assigned_build_queue:
                continue
            if len(nexus.energy) < 150:
                nexus.add_task(nexus.chrono_boost(nexus))

    def expand(self):
        if len(self.units(Nexus)) < 3:
            for nexus in self.known_units(Nexus):
                if nexus.owner != self.ally and not nexus.is_completed:
                    self.build_natural(nexus)

    def build_natural(self, enemy_nexus):
        if self.can_afford(Nexus):
            self.build(Nexus, near=enemy_nexus)

    def build_structures(self):
        for nexus in self.units(Nexus):
            if len(nexus.assigned_build_queue) < 2:
                if len(nexus.energy) > 150:
                    if len(self.units(Pylon)) < self.supply_left // 2:
                        self.build(Pylon, near=nexus)
                    if len(self.units(Nexus)) < 3:
                        self.build(CyberneticsCore, near=nexus)
                    if len(self.units(Nexus)) > 1 and nexus.mineral_patch not in [p.position for p in self.mineral_patches]:
                        self.build(Assimilator, near=nexus.mineral_patch)
                    if len(self.units(Nexus)) > 1 and nexus.gas_geyser not in [g.position for g in self.gas_geysers]:
                        self.build(Assimilator, near=nexus.gas_geyser)
                    if len(self.units(Gateway)) < 1:
                        self.build(Gateway, near=nexus)
                    if len(self.units(Stargate)) < 3 and len(self.units(Nexus)) >= 3:
                        self.build(Stargate, near=nexus)

    def train_units(self):
        for stargate in self.units(Stargate):
            if len(self.units(Stargate)) >= 3:
                return
            if stargate.is_idle and self.can_afford(VoidRay):
                self.build(VoidRay, stargate)

        for nexus in self.units(Nexus):
            if len(nexus.assigned_build_queue) < 1:
                if len(self.units(Probe)) < self.supply_left:
                    self.build(Probe, near=nexus)

    def attack(self):
        if len(self.units(Nexus)) == 0:
            for worker in self.units(Probe):
                worker.attack(self.enemy_start_locations[0])
        else:
            for unit in self.units(VoidRay):
                if len(self.units(VoidRay)) > 5:
                    target = self.find_closest(self.enemy.units_in_range(unit, 12).filter(lambda u: u.is_attack_unit))
                    if target:
                        unit.attack(target)
                else:
                    target = self.find_closest(self.enemy.townhalls)
                    if target:
                        unit.attack(target)

if __name__ == "__main__":
    ProtossBot.run_game(
        map_name="CatalystLE",
        map_size=(Race.Protoss, Difficulty.Easy),
        initial_units=[UnitTypeId.Nexus],
        initial_minerals=1000,
        initial_gas=100,
        replay_path="replay.SC2Replay",
        visualize=True,
        step_mode=True
    )
```

This code creates a bot that plays as Protoss and follows the given strategy. It builds three bases, three stargates, manages resources, builds structures, trains units, and engages in combat. However, it is a simplified version and might need adjustments to work properly.