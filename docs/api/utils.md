# `causalig.utils`

```{eval-rst}
.. module:: causalig.utils
```

## Graph utilities

All functions take an `adjacency_matrix` with `adjacency_matrix[i, j] != 0` meaning `i -> j`.

```{eval-rst}
.. autofunction:: causalig.utils.find_ancestors

.. autofunction:: causalig.utils.find_eldest_ancestors

.. autofunction:: causalig.utils.graph.get_connected_groups

.. autofunction:: causalig.utils.graph.propagate_upstream
```

## Logging

```{eval-rst}
.. autofunction:: causalig.utils.setup_experiment_logger

.. autofunction:: causalig.utils.get_experiment_logger
```
