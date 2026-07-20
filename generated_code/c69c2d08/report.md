# Benchmark Report

## Run Status

**Failed**

- Run ID: `c69c2d08`
- Failed stage: `artifact collection`
- Error type: `ValidationError`

## Error

1 validation error for BenchmarkResult
task_name
  Field required [type=missing, input_value={'model_name': 'linear_re...r': 0.16115528793543787}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing

## Traceback

```text
Traceback (most recent call last):
  File "/app/main.py", line 97, in main
    BenchmarkResult(model_name=name, **metrics)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/pydantic/main.py", line 263, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
pydantic_core._pydantic_core.ValidationError: 1 validation error for BenchmarkResult
task_name
  Field required [type=missing, input_value={'model_name': 'linear_re...r': 0.16115528793543787}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
```

Partial artifacts generated before the failure may still be present in this run directory.
