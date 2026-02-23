"""Tests for advanced skill functions."""

from __future__ import annotations

import pytest

from piano.core.types import BridgeCommand
from piano.skills.advanced import (
    ADVANCED_SKILLS,
    CraftingRecipes,
    attack_entity,
    build_structure,
    craft_chain,
    defend,
    deposit_items,
    explore_direction,
    farm_harvest,
    farm_plant,
    flee,
    register_advanced_skills,
    smelt_item,
    withdraw_items,
)
from piano.skills.registry import SkillRegistry


class TestCraftingRecipes:
    def test_get_recipe_success(self) -> None:
        recipe = CraftingRecipes.get_recipe("wooden_pickaxe")
        assert recipe == {"planks": 3, "stick": 2}

    def test_get_recipe_not_found(self) -> None:
        with pytest.raises(KeyError):
            CraftingRecipes.get_recipe("nonexistent_item")

    def test_has_recipe_true(self) -> None:
        assert CraftingRecipes.has_recipe("stone_pickaxe")

    def test_has_recipe_false(self) -> None:
        assert not CraftingRecipes.has_recipe("unknown_item")

    def test_recipes_completeness(self) -> None:
        # Check that we have a good set of recipes
        assert len(CraftingRecipes.RECIPES) >= 15
        # Check for essential items
        assert "planks" in CraftingRecipes.RECIPES
        assert "stick" in CraftingRecipes.RECIPES
        assert "wooden_pickaxe" in CraftingRecipes.RECIPES
        assert "stone_pickaxe" in CraftingRecipes.RECIPES
        assert "iron_pickaxe" in CraftingRecipes.RECIPES
        assert "furnace" in CraftingRecipes.RECIPES
        assert "chest" in CraftingRecipes.RECIPES

    def test_recipe_correctness(self) -> None:
        # Verify some specific recipes are correct
        assert CraftingRecipes.get_recipe("planks") == {"log": 1}
        assert CraftingRecipes.get_recipe("stick") == {"planks": 2}
        assert CraftingRecipes.get_recipe("furnace") == {"cobblestone": 8}
        assert CraftingRecipes.get_recipe("chest") == {"planks": 8}
        assert CraftingRecipes.get_recipe("torch") == {"stick": 1, "coal": 1}


class TestCraftChain:
    def test_craft_chain_no_recipe(self) -> None:
        # Base materials with no recipe return empty list
        commands = craft_chain("dirt", {})
        assert commands == []

    def test_craft_chain_simple_item(self) -> None:
        # Crafting planks from logs (already have logs)
        commands = craft_chain("planks", {"log": 5})
        assert len(commands) == 1
        assert commands[0].action == "craft"
        assert commands[0].params["item_name"] == "planks"

    def test_craft_chain_two_step(self) -> None:
        # Crafting sticks requires planks, which requires logs
        commands = craft_chain("stick", {"log": 1})
        # Should craft planks first, then sticks
        assert len(commands) >= 2
        assert any(cmd.params.get("item_name") == "planks" for cmd in commands)
        assert commands[-1].params["item_name"] == "stick"

    def test_craft_chain_complex_item(self) -> None:
        # Crafting wooden pickaxe requires planks and sticks
        commands = craft_chain("wooden_pickaxe", {"log": 2})
        assert len(commands) >= 1
        # Final command should be crafting the pickaxe
        assert commands[-1].action == "craft"
        assert commands[-1].params["item_name"] == "wooden_pickaxe"

    def test_craft_chain_with_partial_inventory(self) -> None:
        # Already have some materials
        inventory = {"planks": 3, "stick": 1}
        commands = craft_chain("wooden_pickaxe", inventory)
        # Should only need to craft 1 more stick
        assert len(commands) >= 1
        assert commands[-1].params["item_name"] == "wooden_pickaxe"

    def test_craft_chain_returns_bridge_commands(self) -> None:
        commands = craft_chain("stone_sword", {"cobblestone": 5, "log": 2})
        assert all(isinstance(cmd, BridgeCommand) for cmd in commands)
        assert all(cmd.action == "craft" for cmd in commands)


class TestBuildStructure:
    def test_build_structure_single_block(self) -> None:
        position = {"x": 0.0, "y": 64.0, "z": 0.0}
        blocks = [{"offset_x": 0, "offset_y": 0, "offset_z": 0, "block_type": "stone"}]
        commands = build_structure("wall", position, blocks)

        assert len(commands) == 1
        assert commands[0].action == "place"
        assert commands[0].params["x"] == 0.0
        assert commands[0].params["y"] == 64.0
        assert commands[0].params["z"] == 0.0
        assert commands[0].params["block_type"] == "stone"

    def test_build_structure_multiple_blocks(self) -> None:
        position = {"x": 10.0, "y": 60.0, "z": -5.0}
        blocks = [
            {"offset_x": 0, "offset_y": 0, "offset_z": 0, "block_type": "stone"},
            {"offset_x": 1, "offset_y": 0, "offset_z": 0, "block_type": "stone"},
            {"offset_x": 0, "offset_y": 1, "offset_z": 0, "block_type": "cobblestone"},
        ]
        commands = build_structure("tower", position, blocks)

        assert len(commands) == 3
        assert commands[0].params["x"] == 10.0
        assert commands[1].params["x"] == 11.0
        assert commands[2].params["y"] == 61.0

    def test_build_structure_returns_bridge_commands(self) -> None:
        position = {"x": 0.0, "y": 0.0, "z": 0.0}
        blocks = [{"offset_x": 0, "offset_y": 0, "offset_z": 0, "block_type": "dirt"}]
        commands = build_structure("test", position, blocks)

        assert all(isinstance(cmd, BridgeCommand) for cmd in commands)
        assert all(cmd.action == "place" for cmd in commands)


class TestFarmSkills:
    def test_farm_plant_command(self) -> None:
        position = {"x": 5.0, "y": 64.0, "z": 10.0}
        cmd = farm_plant("wheat_seeds", position)

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "plant"
        assert cmd.params["crop"] == "wheat_seeds"
        assert cmd.params["x"] == 5.0
        assert cmd.params["y"] == 64.0
        assert cmd.params["z"] == 10.0

    def test_farm_harvest_command(self) -> None:
        position = {"x": 5.0, "y": 64.0, "z": 10.0}
        cmd = farm_harvest(position)

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "harvest"
        assert cmd.params["x"] == 5.0
        assert cmd.params["y"] == 64.0
        assert cmd.params["z"] == 10.0

    def test_farm_plant_different_crops(self) -> None:
        position = {"x": 0.0, "y": 0.0, "z": 0.0}
        crops = ["wheat_seeds", "carrot", "potato", "beetroot_seeds"]

        for crop in crops:
            cmd = farm_plant(crop, position)
            assert cmd.params["crop"] == crop


class TestSmeltItem:
    def test_smelt_item_default_fuel(self) -> None:
        cmd = smelt_item("iron_ore")

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "smelt"
        assert cmd.params["input_item"] == "iron_ore"
        assert cmd.params["fuel"] == "coal"
        assert cmd.params["count"] == 1

    def test_smelt_item_custom_fuel(self) -> None:
        cmd = smelt_item("sand", fuel="lava_bucket", count=64)

        assert cmd.params["input_item"] == "sand"
        assert cmd.params["fuel"] == "lava_bucket"
        assert cmd.params["count"] == 64

    def test_smelt_item_different_materials(self) -> None:
        materials = ["iron_ore", "gold_ore", "sand", "cobblestone"]
        for material in materials:
            cmd = smelt_item(material)
            assert cmd.params["input_item"] == material


class TestExploreDirection:
    def test_explore_direction_default_distance(self) -> None:
        cmd = explore_direction("north")

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "explore"
        assert cmd.params["direction"] == "north"
        assert cmd.params["distance"] == 50.0

    def test_explore_direction_custom_distance(self) -> None:
        cmd = explore_direction("south", distance=100.0)

        assert cmd.params["direction"] == "south"
        assert cmd.params["distance"] == 100.0

    def test_explore_all_directions(self) -> None:
        directions = ["north", "south", "east", "west"]
        for direction in directions:
            cmd = explore_direction(direction)
            assert cmd.params["direction"] == direction


class TestCombatSkills:
    def test_attack_entity_command(self) -> None:
        cmd = attack_entity("zombie")

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "attack"
        assert cmd.params["target"] == "zombie"

    def test_defend_default_shield(self) -> None:
        cmd = defend()

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "defend"
        assert cmd.params["shield"] is True

    def test_defend_no_shield(self) -> None:
        cmd = defend(shield=False)

        assert cmd.params["shield"] is False

    def test_flee_command(self) -> None:
        from_pos = {"x": 10.0, "y": 64.0, "z": -5.0}
        cmd = flee(from_pos, distance=30.0)

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "flee"
        assert cmd.params["from_x"] == 10.0
        assert cmd.params["from_y"] == 64.0
        assert cmd.params["from_z"] == -5.0
        assert cmd.params["distance"] == 30.0

    def test_flee_default_distance(self) -> None:
        from_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        cmd = flee(from_pos)

        assert cmd.params["distance"] == 20.0


class TestStorageManagement:
    def test_deposit_items_command(self) -> None:
        chest_pos = {"x": 100.0, "y": 64.0, "z": 200.0}
        items = {"diamond": 5, "iron_ingot": 32}
        cmd = deposit_items(chest_pos, items)

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "deposit"
        assert cmd.params["x"] == 100.0
        assert cmd.params["y"] == 64.0
        assert cmd.params["z"] == 200.0
        assert cmd.params["items"] == items

    def test_deposit_items_single_item(self) -> None:
        chest_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        items = {"gold_ingot": 1}
        cmd = deposit_items(chest_pos, items)

        assert cmd.params["items"] == {"gold_ingot": 1}

    def test_withdraw_items_command(self) -> None:
        chest_pos = {"x": 100.0, "y": 64.0, "z": 200.0}
        items = ["sword", "pickaxe", "food"]
        cmd = withdraw_items(chest_pos, items)

        assert isinstance(cmd, BridgeCommand)
        assert cmd.action == "withdraw"
        assert cmd.params["x"] == 100.0
        assert cmd.params["y"] == 64.0
        assert cmd.params["z"] == 200.0
        assert cmd.params["items"] == items

    def test_withdraw_items_single_item(self) -> None:
        chest_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        items = ["diamond_sword"]
        cmd = withdraw_items(chest_pos, items)

        assert cmd.params["items"] == ["diamond_sword"]


class TestAdvancedSkillsDict:
    def test_all_skills_in_dict(self) -> None:
        # Check that all expected skills are in the dict
        expected_skills = [
            "craft_chain",
            "build_structure",
            "farm_plant",
            "farm_harvest",
            "smelt_item",
            "explore_direction",
            "attack_entity",
            "defend",
            "flee",
            "deposit_items",
            "withdraw_items",
        ]
        for skill in expected_skills:
            assert skill in ADVANCED_SKILLS

    def test_all_skills_are_callable(self) -> None:
        for name, skill_fn in ADVANCED_SKILLS.items():
            assert callable(skill_fn), f"Skill {name} is not callable"

    def test_skill_dict_completeness(self) -> None:
        # Verify we have exactly 11 skills
        assert len(ADVANCED_SKILLS) == 11


class TestRegisterAdvancedSkills:
    def test_register_advanced_skills_no_error(self) -> None:
        # Currently this is a no-op, but test it doesn't raise
        registry = SkillRegistry()
        register_advanced_skills(registry)
        # No assertion needed, just verify no exception


class TestIntegrationScenarios:
    def test_complete_tool_crafting_workflow(self) -> None:
        # Simulate crafting a stone pickaxe from scratch
        inventory = {"log": 3, "cobblestone": 10}
        commands = craft_chain("stone_pickaxe", inventory)

        # Should produce commands to craft necessary items
        assert len(commands) >= 1
        assert isinstance(commands[-1], BridgeCommand)
        assert commands[-1].params["item_name"] == "stone_pickaxe"

    def test_building_wall_structure(self) -> None:
        # Build a simple 3-block wall
        position = {"x": 0.0, "y": 64.0, "z": 0.0}
        blocks = [
            {"offset_x": i, "offset_y": 0, "offset_z": 0, "block_type": "stone"} for i in range(3)
        ]
        commands = build_structure("wall", position, blocks)

        assert len(commands) == 3
        for i, cmd in enumerate(commands):
            assert cmd.params["x"] == float(i)

    def test_farming_cycle(self) -> None:
        # Plant and harvest
        position = {"x": 5.0, "y": 64.0, "z": 10.0}
        plant_cmd = farm_plant("wheat_seeds", position)
        harvest_cmd = farm_harvest(position)

        assert plant_cmd.action == "plant"
        assert harvest_cmd.action == "harvest"
        # Both should operate on same position
        assert plant_cmd.params["x"] == harvest_cmd.params["x"]

    def test_combat_sequence(self) -> None:
        # Attack, defend, and flee sequence
        enemy_pos = {"x": 10.0, "y": 64.0, "z": -5.0}

        attack_cmd = attack_entity("skeleton")
        defend_cmd = defend(shield=True)
        flee_cmd = flee(enemy_pos, distance=25.0)

        assert attack_cmd.action == "attack"
        assert defend_cmd.action == "defend"
        assert flee_cmd.action == "flee"

    def test_storage_workflow(self) -> None:
        # Deposit and withdraw from chest
        chest_pos = {"x": 100.0, "y": 64.0, "z": 200.0}

        deposit_cmd = deposit_items(chest_pos, {"diamond": 10})
        withdraw_cmd = withdraw_items(chest_pos, ["diamond"])

        assert deposit_cmd.action == "deposit"
        assert withdraw_cmd.action == "withdraw"
        # Both should operate on same chest
        assert deposit_cmd.params["x"] == withdraw_cmd.params["x"]
