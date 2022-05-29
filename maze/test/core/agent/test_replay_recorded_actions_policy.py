"""Contains a unit tests for the replay recorded actions policy."""
from maze.core.agent.heuristic_lunar_lander_policy import HeuristicLunarLanderPolicy
from maze.core.agent.replay_recorded_actions_policy import ReplayRecordedActionsPolicy
from maze.core.wrappers.action_recording_wrapper import ActionRecordingWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv


def test_replay_recorded_actions_policy():
    """unit tests"""

    teacher_policy = HeuristicLunarLanderPolicy()

    env = GymMazeEnv("LunarLander-v2")
    env = ActionRecordingWrapper.wrap(env, record_maze_actions=False, record_actions=True,
                                      output_dir="action_records")

    env.seed(1234)
    obs = env.reset()
    done = False
    cum_reward_teacher = 0
    while not done:
        action = teacher_policy.compute_action(obs)
        obs, rew, done, info = env.step(action)
        cum_reward_teacher += rew

    env.dump()

    episode_id = env.get_episode_id()
    expected_file_path = "action_records/" + str(episode_id) + ".pkl"
    replay_policy = ReplayRecordedActionsPolicy(action_record_path=expected_file_path, with_agent_actions=True)

    env.seed(1234)
    obs = env.reset()
    done = False
    cum_reward_replay = 0
    while not done:
        action = replay_policy.compute_action(obs, maze_state=env.get_maze_state(), env=env)
        obs, rew, done, info = env.step(action)
        cum_reward_replay += rew

    assert cum_reward_teacher == cum_reward_replay
    assert cum_reward_replay == replay_policy.action_record.cum_action_record_reward
