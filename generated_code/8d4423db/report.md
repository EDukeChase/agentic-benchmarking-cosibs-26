# Benchmark Report

## Run Status

**Failed**

- Run ID: `8d4423db`
- Failed stage: `code generation`
- Error type: `APIConnectionError`

## Error

Connection error.

## Traceback

```text
Traceback (most recent call last):
  File "/usr/local/lib/python3.14/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "/usr/local/lib/python3.14/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/usr/local/lib/python3.14/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/usr/local/lib/python3.14/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "/usr/local/lib/python3.14/site-packages/httpcore/_sync/connection.py", line 101, in handle_request
    raise exc
  File "/usr/local/lib/python3.14/site-packages/httpcore/_sync/connection.py", line 78, in handle_request
    stream = self._connect(request)
  File "/usr/local/lib/python3.14/site-packages/httpcore/_sync/connection.py", line 124, in _connect
    stream = self._network_backend.connect_tcp(**kwargs)
  File "/usr/local/lib/python3.14/site-packages/httpcore/_backends/sync.py", line 207, in connect_tcp
    with map_exceptions(exc_map):
         ~~~~~~~~~~~~~~^^^^^^^^^
  File "/usr/local/lib/python3.14/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ConnectError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.14/site-packages/openai/_base_client.py", line 1037, in request
    response = self._send_request(
        request,
        stream=stream or self._should_stream_response_body(request=request),
        **kwargs,
    )
  File "/usr/local/lib/python3.14/site-packages/openai/_client.py", line 478, in _send_request
    response = self._send_with_auth_retry(request, stream=stream, **kwargs)
  File "/usr/local/lib/python3.14/site-packages/openai/_client.py", line 456, in _send_with_auth_retry
    response = super()._send_request(request, stream=stream, **kwargs)
  File "/usr/local/lib/python3.14/site-packages/openai/_base_client.py", line 964, in _send_request
    return self._client.send(request, stream=stream, **kwargs)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "/usr/local/lib/python3.14/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "/usr/local/lib/python3.14/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/usr/local/lib/python3.14/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/usr/local/lib/python3.14/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/usr/local/lib/python3.14/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/main.py", line 82, in main
    run_programming_agent(prog_agent, literature_result)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/agents/programming_agent.py", line 77, in run_programming_agent
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
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 1450, in model_node
    result = wrap_model_call_handler(request, _execute_model_sync)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 311, in composed
    outer_result = outer(request, inner_handler)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/middleware/todo.py", line 256, in wrap_model_call
    return handler(request.override(system_message=new_system_message))
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 301, in inner_handler
    inner_result = inner(req, handler)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 311, in composed
    outer_result = outer(request, inner_handler)
  File "/usr/local/lib/python3.14/site-packages/deepagents/middleware/filesystem.py", line 1937, in wrap_model_call
    return handler(request)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 301, in inner_handler
    inner_result = inner(req, handler)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 311, in composed
    outer_result = outer(request, inner_handler)
  File "/usr/local/lib/python3.14/site-packages/deepagents/middleware/subagents.py", line 856, in wrap_model_call
    return handler(request.override(system_message=new_system_message))
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 301, in inner_handler
    inner_result = inner(req, handler)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 311, in composed
    outer_result = outer(request, inner_handler)
  File "/usr/local/lib/python3.14/site-packages/deepagents/middleware/summarization.py", line 1434, in wrap_model_call
    return handler(request.override(messages=truncated_messages))
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 301, in inner_handler
    inner_result = inner(req, handler)
  File "/usr/local/lib/python3.14/site-packages/langchain_anthropic/middleware/prompt_caching.py", line 168, in wrap_model_call
    return handler(request)
  File "/usr/local/lib/python3.14/site-packages/langchain/agents/factory.py", line 1419, in _execute_model_sync
    output = model_.invoke(messages)
  File "/usr/local/lib/python3.14/site-packages/langchain_core/runnables/base.py", line 6002, in invoke
    return self.bound.invoke(
           ~~~~~~~~~~~~~~~~~^
        input,
        ^^^^^^
        self._merge_configs(config),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        **{**self.kwargs, **kwargs},
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.14/site-packages/langchain_core/language_models/chat_models.py", line 476, in invoke
    self.generate_prompt(
    ~~~~~~~~~~~~~~~~~~~~^
        [self._convert_input(input)],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **kwargs,
        ^^^^^^^^^
    ).generations[0][0],
    ^
  File "/usr/local/lib/python3.14/site-packages/langchain_core/language_models/chat_models.py", line 1849, in generate_prompt
    return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/langchain_core/language_models/chat_models.py", line 1656, in generate
    self._generate_with_cache(
    ~~~~~~~~~~~~~~~~~~~~~~~~~^
        m,
        ^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.14/site-packages/langchain_core/language_models/chat_models.py", line 1994, in _generate_with_cache
    result = self._generate(
        messages, stop=stop, run_manager=run_manager, **kwargs
    )
  File "/usr/local/lib/python3.14/site-packages/langchain_openai/chat_models/base.py", line 1692, in _generate
    _handle_openai_api_error(e)
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/usr/local/lib/python3.14/site-packages/langchain_openai/chat_models/base.py", line 1687, in _generate
    raw_response = self.client.with_raw_response.create(**payload)
  File "/usr/local/lib/python3.14/site-packages/openai/_legacy_response.py", line 367, in wrapped
    return cast(LegacyAPIResponse[R], func(*args, **kwargs))
                                      ~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/openai/_utils/_utils.py", line 298, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.14/site-packages/openai/resources/chat/completions/completions.py", line 1281, in create
    return self._post(
           ~~~~~~~~~~^
        "/chat/completions",
        ^^^^^^^^^^^^^^^^^^^^
    ...<53 lines>...
        stream_cls=Stream[ChatCompletionChunk],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.14/site-packages/openai/_base_client.py", line 1332, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/site-packages/openai/_base_client.py", line 1072, in request
    raise APIConnectionError(request=request) from err
openai.APIConnectionError: Connection error.
During task with name 'model' and id '9d902d3a-f464-9fda-c821-62de7af5ea85'
```

Partial artifacts generated before the failure may still be present in this run directory.
