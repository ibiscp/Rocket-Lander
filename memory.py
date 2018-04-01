import random
from collections import deque

class Memory:

    # Initialize memory
    def __init__(self, capacity):
        self.samples = []# deque(maxlen=capacity)
        self.capacity = capacity

    # Add sample to the memory
    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    # Generate 'n' random samples from the memory
    def sample(self, n):
        #n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    # Number of current samples in memory
    def numberSamples(self):
        return len(self.samples)
