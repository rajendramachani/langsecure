
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


def create_input_check_step(klass):

    @step(workflow=klass)
    async def langsecure_input_check(ev: StartEvent) -> StartEvent | StopEvent:
        # handle StartEvent only when involved through the send_event by step name.
        if not hasattr(ev, "langsecure_shield"):
            return

        shield = ev.langsecure_shield

        query = ev.query
        print(" - Applying input enforcer on query: %s.." % query)
        deny, deny_message = await shield._input_enforcer(query)
        print(" - Input enforcer result: deny: %s, deny_message: %s" % (deny, deny_message))
        if deny:
            # stop workflow
            return StopEvent(result=deny_message)
        else:
            return ev

    return langsecure_input_check


def create_output_check_step(klass):

    @step(workflow=klass)
    async def langsecure_output_check(ev: StopEvent) -> StopEvent:
        # handle StopEvent only when involved through the send_event by step name.
        if not hasattr(ev, "langsecure_shield"):
            return

        shield = ev.langsecure_shield

        result = ev.result
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

          if type(message) is StartEvent:
              if "langsecure_shield" in message._data.keys(): 
                  del message._data["langsecure_shield"]
              else:
                  message._data.update({"langsecure_shield": self})
                  # call by specific step name to avoid task for actual step StartEvent not unblock
                  return parent_callable(message, step="langsecure_input_check", **kwargs)

                        
          if type(message) is StopEvent:
              if "langsecure_shield" in message._data.keys(): 
                  del message._data["langsecure_shield"]
              else:
                  message._data.update({"langsecure_shield": self})
                  # call by specific step name to avoid task for actual step StopEvent not unblock
                  return parent_callable(message, step="langsecure_output_check", **kwargs)

          return parent_callable(message, **kwargs)

      return send_event

