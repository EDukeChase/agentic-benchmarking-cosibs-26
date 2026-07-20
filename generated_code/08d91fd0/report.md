# Benchmark Report

## Run Status

**Failed**

- Run ID: `08d91fd0`
- Failed stage: `benchmarking`
- Error type: `StageTimeoutError`

## Error

Stage 'benchmarking' exceeded its 120-second time limit

## Traceback

```text
Traceback (most recent call last):
  File "/app/main.py", line 91, in main
    run_benchmarking_agent(bench_agent, run_id, literature_result)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/agents/benchmarking_agent.py", line 88, in run_benchmarking_agent
    agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/langgraph/pregel/main.py", line 3913, in invoke
    for chunk in self.stream(
                 ~~~~~~~~~~~^
        input,
        ^^^^^^
    ...<11 lines>...
        **kwargs,
        ^^^^^^^^^
    ):
    ^
  File "/usr/local/lib/python3.14/site-packages/langgraph/pregel/main.py", line 2967, in stream
    for _ in runner.tick(
             ~~~~~~~~~~~^
        [t for t in loop.tasks.values() if not t.writes],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        schedule_task=loop.accept_push,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/usr/local/lib/python3.14/site-packages/langgraph/pregel/_runner.py", line 207, in tick
    run_with_retry(
    ~~~~~~~~~~~~~~^
        t,
        ^^
    ...<10 lines>...
        },
        ^^
    )
    ^
  File "/usr/local/lib/python3.14/site-packages/langgraph/pregel/_retry.py", line 617, in run_with_retry
    return task.proc.invoke(task.input, config)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/langgraph/_internal/_runnable.py", line 684, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/usr/local/lib/python3.14/site-packages/langgraph/_internal/_runnable.py", line 426, in invoke
    ret = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.14/site-packages/langgraph/prebuilt/tool_node.py", line 822, in _func
    outputs = list(
        executor.map(self._run_one, tool_calls, input_types, tool_runtimes)
    )
  File "/usr/local/lib/python3.14/concurrent/futures/_base.py", line 645, in result_iterator
    yield _result_or_cancel(fs.pop())
          ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/usr/local/lib/python3.14/concurrent/futures/_base.py", line 312, in _result_or_cancel
    return fut.result(timeout)
           ~~~~~~~~~~^^^^^^^^^
  File "/usr/local/lib/python3.14/concurrent/futures/_base.py", line 449, in result
    self._condition.wait(timeout)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/usr/local/lib/python3.14/threading.py", line 369, in wait
    waiter.acquire()
    ~~~~~~~~~~~~~~^^
  File "/app/main.py", line 37, in _raise_timeout
    raise StageTimeoutError(f"Stage '{stage}' exceeded its {seconds}-second time limit")
StageTimeoutError: Stage 'benchmarking' exceeded its 120-second time limit
During task with name 'tools' and id 'fd7d7620-d26a-8bf6-8019-f7550aa57290'
```

Partial artifacts generated before the failure may still be present in this run directory.
