# Simple Blackjack-like environment where the agent observes a number, and can choose to get another 'card' (a random from 1-10) or hold
class Env(object):

    def __init__(self):
        # Initialized with a random int from 1-10
        self.state = random.randint(1, 10)

    def perform_action(self, action):
        if action == 0:
            # Choose to hold, game ends and reward is the state
            reward = self.state
            self.state = None
        elif action == 1:
            # Choose to hit, no reward yet, and state is incremented by a random int. If the state exceeds 21, game ends and reward is 0
            reward = 0
            self.state += random.randint(1, 10)
            if self.state > 21:
                return 0, None, True
        else:
            raise ('Bad action')

        done = self.state is None

        return reward, self.state, done

    def reset(self):
        # Resets the game
        self.state = random.randint(1, 10)

    def observe(self):
        # Returns the state
        return self.state


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    # Set up memory storage
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # Simple neural net consisting only of fully connected linear layers. Input is a 1x1 tensor and output is a 1x2 tensor representing the expected discounted value of choosing each action
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x