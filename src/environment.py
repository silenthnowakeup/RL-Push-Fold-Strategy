import numpy as np
from config import STACK_MIN, STACK_MAX, SB, BB


class PokerEnvironment:
    def __init__(self, initial_stack=STACK_MIN, max_stack=STACK_MAX, sb=SB, bb=BB):
        self.initial_stack = initial_stack
        self.max_stack = max_stack
        self.sb = sb
        self.bb = bb
        self.reset()

    def reset(self):
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.sb_stack = np.random.randint(self.initial_stack, self.max_stack + 1)
        self.bb_stack = np.random.randint(self.initial_stack, self.max_stack + 1)
        self.sb_card = self.deck[0]
        self.bb_card = self.deck[1]
        self.pot = self.sb + self.bb
        self.current_player = 0  # 0 for SB, 1 for BB
        self.done = False
        self.info = {}

        return self._get_state()

    def _get_state(self):
        return [self.sb_stack, self.bb_stack, self.sb_card, self.bb_card, self.current_player]

    def step(self, action):
        reward = 0

        if self.current_player == 0:  # SB turn
            if action == 0:  # SB folds
                self.sb_stack -= self.sb
                self.bb_stack += self.pot
                reward = -self.sb
                self.done = True
                self.info['winner'] = 'BB'
            elif action == 1:  # SB pushes
                self.pot += self.sb_stack - self.sb
                self.current_player = 1
                reward = 0
        else:  # BB turn
            if action == 0:  # BB folds
                self.bb_stack -= self.bb
                self.sb_stack += self.pot
                reward = self.pot
                self.done = True
                self.info['winner'] = 'SB'
            elif action == 1:  # BB calls
                self.pot += self.bb_stack - self.bb
                reward = self._compare_hands()
                self.done = True

        state = self._get_state()
        return state, reward, self.done, self.info

    def _compare_hands(self):
        if self.sb_card // 4 > self.bb_card // 4:
            self.sb_stack += self.pot
            self.info['winner'] = 'SB'
            return self.pot
        elif self.bb_card // 4 > self.sb_card // 4:
            self.bb_stack += self.pot
            self.info['winner'] = 'BB'
            return -self.pot
        else:
            self.sb_stack += self.pot // 2
            self.bb_stack += self.pot // 2
            self.info['winner'] = 'Draw'
            return 0
