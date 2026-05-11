class StackedRNN:
    def __init__(self, layers):
        if not layers:
            raise ValueError("StackedRNN requires at least one recurrent layer")
        self.layers = layers

    def forward(self, x, initial_state=None):
        out = x
        states = []

        for index, layer in enumerate(self.layers):
            state = None
            if initial_state is not None:
                state = initial_state[index]

            result = layer.forward(out, initial_state=state)
            if isinstance(result, tuple):
                out = result[0]
                states.append(result[1:])
            else:
                out = result

        if states:
            return out, states
        return out
