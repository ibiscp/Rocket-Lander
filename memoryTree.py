import random
from sumtree import SumTree

class Memory:
    # Constants
    e = 0.01
    a = 0.0  #0.6

    # Initialize memory
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.len = 0

    # Calculate error priority
    def getPriority(self, error):
        return (error + self.e) ** self.a

    # Add sample to the memory
    def add(self, error, sample):
        p = self.getPriority(error)
        self.tree.add(p, sample)
        self.len = min(self.len + 1, self.capacity)

    # Generate 'n' random samples from the memory
    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    # Number of current samples in memory
    def numberSamples(self):
        return self.len

    # Update priority of error
    def update(self, idx, error):
        p = self.getPriority(error)
        self.tree.update(idx, p)
