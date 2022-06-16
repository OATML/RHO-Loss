class fixed_schedule:
    def __init__(self, initial_temp):
        self.initial_temp = initial_temp

    def __call__(self, current_epoch):
        return self.initial_temp


class linear_schedule:
    def __init__(self, initial_temp, temp_increment):
        self.initial_temp = initial_temp
        self.temp_increment = temp_increment

    def __call__(self, current_epoch):
        return self.initial_temp + self.temp_increment * current_epoch


class geometric_schedule:
    def __init__(self, initial_temp, increment_factor):
        self.initial_temp = initial_temp
        self.increment_factor = increment_factor

    def __call__(self, current_epoch):
        return self.initial_temp * (self.increment_factor) ** (current_epoch - 1)
