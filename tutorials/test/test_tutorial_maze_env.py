"""Contains tutorial unit tests"""
from tutorial_maze_env.part01_core_env.main import main as main_part01
from tutorial_maze_env.part02_renderer.main import main as main_part02
from tutorial_maze_env.part03_maze_env.main import main as main_part03
from tutorial_maze_env.part04_events.main import main as main_part04
from tutorial_maze_env.part05_reward.main import main as main_part05
from tutorial_maze_env.part06_struct_env.main import main as main_part06


def test_tutorial_main_functions():
    """tutorial testing function"""
    main_part01()
    main_part02()
    main_part03()
    main_part04()
    main_part05()
    main_part06()
