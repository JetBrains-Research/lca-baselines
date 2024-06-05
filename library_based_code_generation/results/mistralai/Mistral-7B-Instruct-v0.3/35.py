 Here is a basic structure for the StarCraft II bot using the python-sc2 library. Note that this is a simplified version and might need further adjustments and improvements.

```python
from sc2 import RunGame, run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.races import Protoss
from sc2.units import Units
from sc2.actions import Function, SelectUnitByDistance
from sc2.data import UnitTypeId

class ProtossBot(Bot):
    async def on_step(self, iteration):
        if not self.alive:
            return False

        # Resource management
        self.distribute_workers()
        self.build_pylons_when_needed()

        # Unit production
        self.train_probes()
        self.build_gateways()
        self.build_gas()
        self.research_warp_gate()
        self.morph_to_warp_gate()
        self.warp_new_units()

        # Combat
        target = self.find_closest_enemy_unit() or self.find_enemy_spawn_location()
        self.make_stalkers_attack(target)

        # Proxy pylon
        self.build_proxy_pylon()

        # Upgrades
        self.chrono_boost(UnitTypeId.Nexus, UnitTypeId.CyberCore)

        # Handle no nexuses left
        if not self.units(UnitTypeId.Nexus).ready.exists:
            self.train_probes()

    def find_closest_enemy_unit(self):
        return self.select_units(Unittypeids.enemy).sorted(key=lambda x: x.distance_to(self.units(UnitTypeId.Stalker)[0]))[0]

    def find_enemy_spawn_location(self):
        # Implement logic to find enemy spawn location
        pass

    def distribute_workers(self):
        # Implement logic to distribute workers
        pass

    def build_pylons_when_needed(self):
        # Implement logic to build pylons when on low supply
        pass

    def train_probes(self):
        # Implement logic to train probes
        pass

    def build_gateways(self):
        # Implement logic to build gateways
        pass

    def build_gas(self):
        # Implement logic to build gas
        pass

    def research_warp_gate(self):
        # Implement logic to research warp gate
        pass

    def morph_to_warp_gate(self):
        # Implement logic to morph to warp gate when research is complete
        pass

    def warp_new_units(self):
        # Implement logic to warp new units
        pass

    def make_stalkers_attack(self, target):
        # Implement logic to make stalkers attack either closest enemy unit or enemy spawn location
        pass

    def build_proxy_pylon(self):
        # Implement logic to build proxy pylon
        pass

    def chrono_boost(self, unit_type_id, building_type_id=None):
        # Implement logic to chrono boost nexus or cybercore
        pass

if __name__ == "__main__":
    run_game(
        map_name=maps.get("2CatalystLE"),
        players=[
            Computer(Race.Protoss, Difficulty.Easy),
            ProtossBot(Race.Protoss),
        ],
        realtime=True,
        step_mode=RunGame.STEPS_PER_MINUTE_30,
    )
```

This code provides a basic structure for the bot, but you will need to implement the methods for each specific behavior. The `find_closest_enemy_unit` and `find_enemy_spawn_location` methods are placeholders and should be replaced with your own logic to find the closest enemy unit and enemy spawn location. Similarly, the other methods need to be implemented according to the desired behavior of your bot.