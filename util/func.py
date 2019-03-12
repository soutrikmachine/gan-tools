

def do_n_times(n, fun, aggregate_func=None, **fun_args):
    return_values = []
    for i in range(n):
        tup = fun(**fun_args)
        return_values.append(list(tup))
    return_values = list(map(list, zip(*return_values)))  # transpose
    print(return_values)
    if aggregate_func is not None:
        aggregated_return_values = []
        for l in return_values:
            aggregated_return_values.append(aggregate_func(l))
        return aggregated_return_values
    return return_values
