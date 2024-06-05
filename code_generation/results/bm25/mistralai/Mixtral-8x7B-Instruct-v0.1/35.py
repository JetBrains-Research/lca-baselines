 from sc2.constants import *
from sc2.position import Point2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2 import Race, Map, Unit, Game, UnitTypeData
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty

class MyBot(BotAI):

def __init__(self, **kwargs):
self.iteration = 0
self.proxy_pylon_built = False

def on_start(self):
self.settings.food_target = 44
self.settings.supply_target = 50
self.settings.expand_threshold = 10000
self.settings.use_mineral_walk = True
self.settings.use_gas_walk = True
self.settings.use_scout_workers = True
self.settings.use_map_information = True
self.settings.use_grid_information = True
self.settings.attack_move_position = Point2(50, 50)
self.settings.use_micro_attacks = True
self.settings.use_multi_prong_attacks = True
self.settings.use_smart_attack_move = True
self.settings.use_scout_attacks = True
self.settings.use_static_defense = True
self.settings.use_tech_build_openings = True
self.settings.use_worker_rallies = True
self.settings.use_worker_scouting = True
self.settings.use_worker_transfer = True
self.settings.use_cheese_detection = True
self.settings.use_base_fighting = True
self.settings.use_base_defense = True
self.settings.use_base_management = True
self.settings.use_production_queue = True
self.settings.use_production_priority = True
self.settings.use_upgrade_priority = True
self.settings.use_unit_counts = True
self.settings.use_unit_composition = True
self.settings.use_unit_combat_priority = True
self.settings.use_unit_formation = True
self.settings.use_unit_grouping = True
self.settings.use_unit_movement = True
self.settings.use_unit_pathfinding = True
self.settings.use_unit_positioning = True
self.settings.use_unit_production = True
self.settings.use_unit_recycling = True
self.settings.use_unit_reinforcement = True
self.settings.use_unit_selection = True
self.settings.use_unit_targeting = True
self.settings.use_unit_upgrades = True
self.settings.use_worker_distribution = True
self.settings.use_worker_distribution_on_minerals = True
self.settings.use_worker_distribution_on_gas = True
self.settings.use_worker_distribution_on_idle_minerals = True
self.settings.use_worker_distribution_on_idle_gas = True
self.settings.use_worker_distribution_on_idle_workers = True
self.settings.use_worker_distribution_on_low_minerals = True
self.settings.use_worker_distribution_on_low_gas = True
self.settings.use_worker_distribution_on_low_supply = True
self.settings.use_worker_distribution_on_no_minerals = True
self.settings.use_worker_distribution_on_no_gas = True
self.settings.use_worker_distribution_on_no_supply = True
self.settings.use_worker_distribution_on_full_minerals = True
self.settings.use_worker_distribution_on_full_gas = True
self.settings.use_worker_distribution_on_full_supply = True
self.settings.use_worker_distribution_on_enemy_minerals = True
self.settings.use_worker_distribution_on_enemy_gas = True
self.settings.use_worker_distribution_on_enemy_supply = True
self.settings.use_worker_distribution_on_enemy_structures = True
self.settings.use_worker_distribution_on_enemy_workers = True
self.settings.use_worker_distribution_on_enemy_units = True
self.settings.use_worker_distribution_on_enemy_structures_and_workers = True
self.settings.use_worker_distribution_on_enemy_structures_and_units = True
self.settings.use_worker_distribution_on_enemy_workers_and_units = True
self.settings.use_worker_distribution_on_enemy_structures_workers_and_units = True
self.settings.use_worker_distribution_on_all_enemy_units = True
self.settings.use_worker_distribution_on_all_enemy_structures = True
self.settings.use_worker_distribution_on_all_enemy_workers = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_idle_minerals = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_idle_gas = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_idle_workers = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_low_minerals = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_low_gas = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_low_supply = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_no_minerals = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_no_gas = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_no_supply = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_full_minerals = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_full_gas = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_full_supply = True
self.settings.use_worker_distribution_on_all_enemy_units_structures_and_workers_and_minerals_and_gas_and_supply_and_idle_gas = True
self.