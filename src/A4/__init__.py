from timeit import default_timer


def timing(f):
    def time_func(*args, **kwargs):
        start_time = default_timer()
        r = f(*args, **kwargs)
        total_time = default_timer() - start_time
        print('Time elapsed: {}'.format(total_time))
        return r
    return time_func


