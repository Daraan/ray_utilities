import gymnasium as gym

env_short_names = {
    "lunar": "LunarLander-v2",
    "cart": "CartPole-v1",
    "cartpole": "CartPole-v1",
}


def create_env(name: str, **kwargs) -> gym.Env:
    if name in env_short_names:
        return gym.make(env_short_names[name], **kwargs)
    return gym.make(name, **kwargs)
