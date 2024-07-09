from agent import Agent
from environment import PokerEnvironment
from config import ACTOR_MODEL_PATH, CRITIC_MODEL_PATH


def play():
    env = PokerEnvironment()
    agent = Agent()
    agent.load_model(ACTOR_MODEL_PATH, CRITIC_MODEL_PATH)

    while True:
        state = env.reset()
        done = False
        print("-" * 20)
        print("New Game Started!")
        print(f"Your Stack (SB): {env.sb_stack}")
        print(f"Agent's Stack (BB): {env.bb_stack}")
        print(f"Your Card: {env.sb_card}")

        while not done:
            if env.current_player == 0:  # SB turn (human)
                action = input("Your Action (0: Fold, 1: Push): ")
                action = int(action)
                state, reward, done, info = env.step(action)
            else:  # BB turn (agent)
                action = agent.get_action(state)
                state, reward, done, info = env.step(action)
                print(f"Agent's Action: {'Call' if action == 1 else 'Fold'}")

            if done:
                print(f"Game Over! Winner: {info['winner']}")
                print(f"Your Stack: {env.sb_stack}, Agent's Stack: {env.bb_stack}, Agent's Card: {env.bb_card}")

        play_again = input("Do you want to play again? (y/n): ").strip().lower()
        if play_again != 'y':
            break


if __name__ == "__main__":
    play()
