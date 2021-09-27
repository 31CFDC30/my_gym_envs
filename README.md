# my_gym_envs

在上级目录中的__init__.py文件中添加以下内容：

        # DIY envs
        # ---------

        register(
            id='GridWorld-v1',
            entry_point='gym.envs.my_gym_envs:GridEnv1',
            max_episode_steps=200,
            reward_threshold=100.0,
            )