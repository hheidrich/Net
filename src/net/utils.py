def net_walker(walker):
    rw_generator = walker.walk()
    while True:
        random_walks = next(rw_generator)
        yield transform_to_transitions(random_walks)

def transform_to_transitions(random_walks):
    x = torch.tensor(random_walks[:, :-1].reshape([-1]))
    y = torch.tensor(random_walks[:, 1:].reshape([-1]))
    return x, y
