import gym


env = gym.make('Breakout-4skips-v0')
env.reset()
done = False
while not done:

    env.render()
    action = input("Enter an action: ")
    _, reward, done, _ = env.step(action)
    print(reward)


