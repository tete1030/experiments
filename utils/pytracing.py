# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import json
import time
import threading
from contextlib import contextmanager

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


def to_microseconds(s):
    return 1000000 * float(s)


class TraceWriter(threading.Thread):

    def __init__(self, terminator, input_queue, output_stream):
        threading.Thread.__init__(self)
        self.daemon = True
        self.terminator = terminator
        self.input = input_queue
        self.output = output_stream

    def run(self):
        while not self.terminator.is_set():
            item = self.input.get()
            self.output.write(item)


class TraceProfiler(object):
    """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).
       Usage:
          from pytracing import TraceProfiler
          tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
          with tp.traced():
            ...
    """
    TYPES = {'call': 'B', 'return': 'E'}

    def __init__(self, output_file, clock=None, enabled=True):
        if enabled:
            self.output = open(output_file, "w")
            self.clock = clock or time.time
            self.pid = os.getpid()
            self.queue = Queue()
            self.terminator = threading.Event()
            self.writer = TraceWriter(self.terminator, self.queue, self.output)

    @property
    def thread_id(self):
        return threading.current_thread().name

    @contextmanager
    def traced(self, enabled=True):
        """Context manager for install/shutdown in a with block."""
        if enabled:
            self.install()
        try:
            yield
        finally:
            if enabled:
                self.shutdown()

    def install(self):
        """Install the trace function and open the JSON output stream."""
        self._open_collection()            # Open the JSON output.
        self.writer.start()                # Start the writer thread.
        sys.setprofile(self.tracer)        # Set the trace/profile function.
        # Set the trace/profile function for threads.
        threading.setprofile(self.tracer)

    def shutdown(self):
        sys.setprofile(None)               # Clear the trace/profile function.
        # Clear the trace/profile function for threads.
        threading.setprofile(None)
        self._close_collection()           # Close the JSON output.
        self.terminator.set()              # Stop the writer thread.
        self.writer.join()                 # Join the writer thread.

    def _open_collection(self):
        """Write the opening of a JSON array to the output."""
        self.queue.put('[\n')

    def _close_collection(self):
        """Write the closing of a JSON array to the output."""
        self.queue.put('{}\n]\n')

    def fire_event(self, event_type, func_name, func_filename, func_line_no,
                   caller_filename, caller_line_no):
        """Write a trace event to the output stream."""
        timestamp = to_microseconds(self.clock())
        # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        event = json.dumps(
            dict(
                name=func_name,               # Event Name.
                cat=func_filename,            # Event Category.
                tid=self.thread_id,           # Thread ID.
                ph=self.TYPES[event_type],    # Event Type.
                pid=self.pid,                 # Process ID.
                ts=timestamp,                 # Timestamp.
                args=dict(
                    function=':'.join(
                        [str(x) for x in (func_filename, func_line_no, func_name)]),
                    caller=':'.join(
                        [str(x) for x in (caller_filename, caller_line_no)]),
                )
            )
        ) + ',\n'
        self.queue.put(event)

    def tracer(self, frame, event_type, arg):
        """Bound tracer function for sys.settrace()."""
        try:
            if event_type in self.TYPES.keys() and frame.f_code.co_name != 'write':
                self.fire_event(
                    event_type=event_type,
                    func_name=frame.f_code.co_name,
                    func_filename=frame.f_code.co_filename,
                    func_line_no=frame.f_lineno,
                    caller_filename=frame.f_back.f_code.co_filename,
                    caller_line_no=frame.f_back.f_lineno,
                )
        except Exception:
            pass     # Don't disturb execution if we can't log the trace.
