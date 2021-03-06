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


class GCSSGoalPathBuffer:
    def __init__(
            self,
            max_buffer_size,
            start_state_dim,
            goal_state_dim,
            skill_dim,
    ):
        self.max_buffer_size = max_buffer_size
        self.start_state_dim = start_state_dim
        self.goal_state_dim = goal_state_dim
        self.skill_dim = skill_dim
        self._start_states = np.zeros((max_buffer_size, start_state_dim))
        self._goal_states = np.zeros((max_buffer_size, goal_state_dim))
        self._skills = np.zeros((max_buffer_size, skill_dim))
        self._top = 0
        self._size = 0

    def add_samples(self, samples):
        for sample in samples:
            start_states = sample['start_states']
            goal_states = sample['final_states']
            skills = sample['skills']
            num_samples = len(start_states)
            assert num_samples <= self.max_buffer_size, 'Number of samples cannot exceed max buffer size.'
            if self._size < self.max_buffer_size:
                self._size = min(self._size + num_samples, self.max_buffer_size)
            if self._top + num_samples <= self.max_buffer_size:
                self._start_states[self._top:self._top+num_samples] = start_states
                self._goal_states[self._top:self._top + num_samples] = goal_states
                self._skills[self._top:self._top + num_samples] = skills
                self._top = (self._top + num_samples) % self.max_buffer_size
            else:
                split = self.max_buffer_size-self._top
                new_top = (self._top + num_samples) % self.max_buffer_size
                self._start_states[self._top:] = start_states[:split]
                self._goal_states[self._top:] = goal_states[:split]
                self._skills[self._top:] = skills[:split]
                self._start_states[:new_top] = start_states[split:]
                self._goal_states[:new_top] = goal_states[split:]
                self._skills[:new_top] = skills[split:]
                self._top = new_top


    def remove(self, num):
        pass

    def random_batch(self, num):
        return_idx = np.random.randint(0, self._size - 1, num)
        return dict(
            start_states=self._start_states[return_idx],
            goal_states=self._goal_states[return_idx],
            skills=self._skills[return_idx]
        )
        # if far_away_from is not None:
        #     num_sample = 50 if self._size > 50 else self._size
        #     indices = np.random.randint(0, self._size-1, num_sample)
        #     candidate = self._goals[indices]
        #     return_idx = np.sum(np.square(candidate - far_away_from), axis=1).argmax()
        # else:
        #     return_idx = np.random.randint(0, self._size-1, 1)
        # return self._goals[return_idx]
