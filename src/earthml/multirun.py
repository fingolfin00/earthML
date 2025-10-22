import toml

class MultiRun:
    def __init__ (self, tomlfn="earthml.toml"):
        self.tomlfn = tomlfn
        self.config = toml.load(tomlfn)

    def generate_multirun (self):
        hyper, keys = [], []
        for key in self.config["hyper"]:
            hyper.append(self.config["hyper"][key])
            keys.append(key)
        # '*' unpacks the list of arrays into separate arguments for product
        all_combinations = itertools.product(*hyper)
        run_configurations = {}
        for i, combo_tuple in enumerate(all_combinations):
            # Use zip to map the parameter keys to their values in the current combination
            run_parameters = dict(zip(keys, combo_tuple))
            run_configurations[f"run_{i}"] = run_parameters
        return run_configurations