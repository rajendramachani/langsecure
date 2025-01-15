
import sys
import asyncio
import inspect
from pathlib import Path
import sys
from typing import Any

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    step,
    Workflow
)
from llama_index.llms.openai import OpenAI

from langsecure import Langsecure
from langsecure.factory import implements


def is_using_asyncio_run(cls, method_name):
    """Check if the given method by name in a class uses asyncio.run()."""
    # Get the method
    method = getattr(cls, method_name)
    
    # Check if the method is implemented in the class
    if not hasattr(method, "__code__"):
        return False

    # Get the source code of the method
    try:
        source_code = inspect.getsource(method)
    except (OSError, TypeError):  # Handles cases where the source isn't available
        return False

    # Check if 'asyncio.run' is in the source code
    return "asyncio.run" in source_code or "asyncio_run" in source_code


def create_input_check_step(klass):

    @step(workflow=klass)
    async def langsecure_input_check(ev: StartEvent) -> StartEvent | StopEvent:
        # handle StartEvent only when invoked through the send_event by step name.
        if not hasattr(ev, "langsecure_shield"):
            return

        shield = ev.langsecure_shield

        query = getattr(ev, shield._input_rail_query_param_name)
        print(" - Applying input enforcer on query: %s.." % query)
        deny, deny_message = await shield._input_enforcer(query)
        print(" - Input enforcer result: deny: %s, deny_message: %s" % (deny, deny_message))
        if deny:
            # stop workflow
            return StopEvent(result=deny_message)
        else:
            return ev

    return langsecure_input_check


def create_context_check_step(klass, accepted_event):

    @step(workflow=klass)
    async def langsecure_context_check(ev: accepted_event) -> accepted_event:
        # handle Event only when invoked through the send_event by step name.
        if not hasattr(ev, "langsecure_shield"):
            return

        shield = ev.langsecure_shield

        nodes = getattr(ev, shield._context_rail_nodes_param_name)
        print(" - Applying context enforcer on nodes: %s.." % nodes)
        # deny, deny_message = await shield._output_enforcer(result)
        # setattr(ev, shield._context_rail_nodes_param_name, nodes)
        # deny, deny_message = False, ""
        # print(" - Output enforcer result: deny: %s, deny_message: %s" % (deny, deny_message))
        # if deny:
        #     # modify result with deny_message
        #     ev.result = deny_message

        return ev
    
    return langsecure_context_check


def create_output_check_step(klass):

    @step(workflow=klass)
    async def langsecure_output_check(ev: StopEvent) -> StopEvent:
        # handle StopEvent only when invoked through the send_event by step name.
        if not hasattr(ev, "langsecure_shield"):
            return

        shield = ev.langsecure_shield

        result = ev.result
        if not is_using_asyncio_run(result.__class__, "__str__"):
            print(" - Applying output enforcer on result: %s.." % result)
        # deny, deny_message = await shield._output_enforcer(result)
        deny, deny_message = False,""
        print(" - Output enforcer result: deny: %s, deny_message: %s" % (deny, deny_message))
        if deny:
            # modify result with deny_message
            ev.result = deny_message

        return ev
    
    return langsecure_output_check


@implements('llama_index.core.workflow.workflow.Workflow')
class LI_Workflow(Langsecure):

    def shield(self, runnable: Any) -> Any:
        self._parent_runnable = runnable
        self._parent_callables_for_runnable = {
            name: func
            for name, func in inspect.getmembers(runnable, predicate=inspect.ismethod)}

        self._parent_runnable._start = self._start
        self._parent_callables_for_ctx = {}

        # add new steps
        create_input_check_step(runnable.__class__)
        create_output_check_step(runnable.__class__)

        self._events_with_retrieved_context = []
        self._context_rail_nodes_param_name = "nodes"
        self._input_rail_query_param_name = "query"
        steps = runnable._get_steps()
        for step_func in steps.values():
            if hasattr(step_func, "_query_param_name"):
                self._input_rail_query_param_name = step_func._query_param_name
            if hasattr(step_func, "_apply_context_rail"):
                step_config = getattr(step_func, "__step_config")
                self._events_with_retrieved_context = step_config.return_types
                self._context_rail_nodes_param_name = step_func._nodes_param_name
                create_context_check_step(runnable.__class__, step_config.return_types[0])
                break

        return self._parent_runnable

    def _start(self, **kwargs):
        """
        _start method of Workflow class
        """
        parent_callable = self._parent_callables_for_runnable["_start"]

        ctx, run_id = parent_callable(**kwargs)

        ctx_obj_id = id(ctx)
        self._parent_callables_for_ctx[ctx_obj_id] = {
            name: func
            for name, func in inspect.getmembers(ctx, predicate=inspect.ismethod)}
        ctx.send_event = self.new_send_event(ctx_obj_id)

        return ctx, run_id

    def new_send_event(self, ctx_obj_id):
      def send_event(message: Event, li_workflow=self, ctx_obj_id=ctx_obj_id, **kwargs):
          """
          send_event method of Context class
          """
          parent_callable = li_workflow._parent_callables_for_ctx[ctx_obj_id]["send_event"]

          if (type(message) is StartEvent and
                hasattr(message, self._input_rail_query_param_name)):
              if "langsecure_shield" in message._data.keys():
                  del message._data["langsecure_shield"]
              else:
                  message._data.update({"langsecure_shield": self})
                  # call by specific step name to avoid task for actual step for StartEvent not unblock
                  return parent_callable(message, step="langsecure_input_check", **kwargs)

          if (type(message) in self._events_with_retrieved_context and
                hasattr(message, self._context_rail_nodes_param_name)):
              if "langsecure_shield" in message._data.keys():
                  del message._data["langsecure_shield"]
              else:
                  message._data.update({"langsecure_shield": self})
                  # call by specific step name to avoid task for actual step not unblock
                  return parent_callable(message, step="langsecure_context_check", **kwargs)

          if type(message) is StopEvent:
              if "langsecure_shield" in message._data.keys():
                  del message._data["langsecure_shield"]
              else:
                  message._data.update({"langsecure_shield": self})
                  # call by specific step name to avoid task for actual step for StopEvent not unblock
                  return parent_callable(message, step="langsecure_output_check", **kwargs)

          return parent_callable(message, **kwargs)

      return send_event
