def chunk(iterable, step):
    starts = range(0, len(iterable) - 1, step)
    stops = map(lambda x: min(x, len(iterable)), range(step, len(iterable) + step, step))
    for start, stop in zip(starts, stops):
        yield start, stop
