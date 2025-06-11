from ray import tune

param_space = {
    "lr": tune.grid_search([0.01, 0.001]),
    "seed": tune.grid_search([0, 1, 2, 3]),
}
