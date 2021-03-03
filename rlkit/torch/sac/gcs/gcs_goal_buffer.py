import numpy as np

class GCSGoalBuffer:
    def __init__(
            self,
            max_buffer_size,
            goal_dim,
    ):
        self.max_buffer_size = max_buffer_size
        self.goal_dim = goal_dim
        self._goals = np.zeros((max_buffer_size, goal_dim))
        self._top = 0
        self._size = 0

    def add(self, samples):
        num_samples = len(samples)
        assert num_samples <= self.max_buffer_size, 'Number of samples cannot exceed max buffer size.'
        if self._size < self.max_buffer_size:
            self._size = min(self._size + num_samples, self.max_buffer_size)
        if self._top + num_samples <= self.max_buffer_size:
            self._goals[self._top:self._top+num_samples] = samples
            self._top = (self._top + num_samples) % self.max_buffer_size
        else:
            split = self.max_buffer_size-self._top-1
            new_top = (self._top + num_samples) % self.max_buffer_size
            self._goals[self._top:] = samples[:split]
            self._goals[:new_top] = samples[split:]
            self._top = new_top


    def remove(self, num):
        pass

    def pick(self, far_away_from=None):
        if far_away_from is not None:
            num_sample = 50 if self._size > 50 else self._size
            indices = np.random.randint(0, self._size-1, num_sample)
            candidate = self._goals[indices]
            return_idx = np.sum(np.square(candidate - far_away_from), axis=1).argmax()
        else:
            return_idx = np.random.randint(0, self._size-1, 1)
        return self._goals[return_idx]

