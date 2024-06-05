 Here is a basic structure for a Zerg rush StarCraft II bot using the python-sc2 library. Please note that this is a simplified version and may need further adjustments and improvements. Also, you need to have the python-sc2 library installed.

```python
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.data import Difficulty
from sc2.managers import MineralFieldManager
from sc2.ids.unittypes import Zerg
from sc2.ids.upgrade_ids import ChitinousPlating
from sc2.ids.ability_id import CAST_BROODLORD
from sc2.ids.unit_actions import RALLY_POINT
from sc2.ids.unit_types import Zerg
from sc2.ids.unit_types import Zerg_Hatchery
from sc2.ids.unit_types import Zerg_Overlord
from sc2.ids.unit_types import Zerg_Drone
from sc2.ids.unit_types import Zerg_Lair
from sc2.ids.unit_types import Zerg_Hydralisk
from sc2.ids.unit_types import Zerg_SpineCrawler
from sc2.ids.unit_types import Zerg_Roach
from sc2.ids.unit_types import Zerg_Queen
from sc2.ids.unit_types import Zerg_Baneling
from sc2.ids.unit_types import Zerg_Infestor
from sc2.ids.unit_types import Zerg_Ultralisk
from sc2.ids.unit_types import Zerg_Broodlord
from sc2.ids.unit_types import Zerg_Spire
from sc2.ids.unit_types import Zerg_Viper
from sc2.ids.unit_types import Zerg_SwarmHost
from sc2.ids.unit_types import Zerg_CreepTumor
from sc2.ids.unit_types import Zerg_CreepColony
from sc2.ids.unit_types import Zerg_CreepSpore
from sc2.ids.unit_types import Zerg_CreepExtractor
from sc2.ids.unit_types import Zerg_CreepGeyser
from sc2.ids.unit_types import Zerg_CreepWall
from sc2.ids.unit_types import Zerg_CreepTumor
from sc2.ids.unit_types import Zerg_CreepColony
from sc2.ids.unit_types import Zerg_CreepSpore
from sc2.ids.unit_types import Zerg_CreepExtractor
from sc2.ids.unit_types import Zerg_CreepGeyser
from sc2.ids.unit_types import Zerg_CreepWall

class ZergRushBot(Bot):
    async def on_step(self, iteration):
        if iteration == 1:
            self.game_step = 2

        # Your bot logic here
        # Send chat message
        self.send_chat("Rush incoming!")

        # Attack enemy structures
        for structure in self.enemy.structures.ready.mineral_fields:
            if structure.is_visible:
                self.do(self.army.attack(structure))

        # Inject hatcheries with larva
        for hatchery in self.units(Zerg_Hatchery):
            if hatchery.can_inject():
                hatchery.inject()

        # Manage vespene gas and mineral resources
        # ...

        # Research upgrades
        if self.can_afford(ChitinousPlating):
            self.build_order.add(ChitinousPlating(self.units(Zerg_Drone)[0]))

        # Train units
        if self.can_afford(Zerg_Hydralisk):
            self.build(Zerg_Hydralisk)

        # Build structures
        if self.can_afford(Zerg_Lair) and not self.already_pending(Zerg_Lair):
            self.build(Zerg_Lair)

    async def on_start(self):
        self.game_step = 1
        self.draw_creep_pixelmap()

    async def on_end(self):
        print("Game has ended.")

    def draw_creep_pixelmap(self):
        # Your creep pixelmap drawing logic here
        pass

def run_bot(map_name, bot):
    run_game(
        map_name=maps.get(map_name),
        players=[
            Computer(Race.Terran, Difficulty.Medium),
            bot
        ],
        realtime=False,
        replay_path="replay.SC2Replay",
    )

if __name__ == "__main__":
    run_bot("AbyssalReefLE", ZergRushBot)
```

This code creates a ZergRushBot that performs a rush strategy. The bot has methods to handle the start, each step, and end of the game. It also has a method to draw a creep pixelmap for debugging purposes. The bot is run on the AbyssalReefLE map against a computer opponent of the Terran race with medium difficulty. The game is not run in real time and a replay of the game is saved.