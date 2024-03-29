# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Writes events to disk in a logdir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import socket
import threading
import time

import six

from tensorboardX.proto import event_pb2
from tensorboardX.record_writer import RecordWriter, directory_check

class EventsWriter(object):
    '''Writes `Event` protocol buffers to an event file.'''

    def __init__(self, file_prefix, filename_suffix='', filename=None, timestamp=None):
        '''
        Events files have a name of the form
        '/some/file/path/events.out.tfevents.[timestamp].[hostname]'
        '''
        if timestamp is None:
            timestamp = int(time.time())
        timestamp = str(timestamp)

        if filename is None:
            self._file_name = file_prefix + ".out.tfevents." + timestamp + "." +\
                socket.gethostname() + filename_suffix
        else:
            self._file_name = filename

        self._num_outstanding_events = 0

        self._py_recordio_writer = RecordWriter(self._file_name)

        # Initialize an event instance.
        self._event = event_pb2.Event()

        self._event.wall_time = timestamp

        self.write_event(self._event)

    def write_event(self, event):
        '''Append "event" to the file.'''

        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        self._num_outstanding_events += 1
        self._py_recordio_writer.write(event_str)

    def flush(self):
        '''Flushes the event file to disk.'''
        self._num_outstanding_events = 0
        return True

    def close(self):
        '''Call self.flush().'''
        return_value = self.flush()
        self._py_recordio_writer.close()
        return return_value


class EventFileWriter(object):
    """Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    @@__init__
    @@add_event
    @@flush
    @@close
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix='', filename=None, timestamp=None):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.
        Args:
          logdir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
        """
        self._logdir = logdir
        directory_check(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = EventsWriter(os.path.join(
            self._logdir, "events"), filename_suffix=filename_suffix, filename=filename, timestamp=timestamp)
        self._closed = False
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                          flush_secs)

        self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        if self._closed:
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        if not self._closed:
            self._event_queue.put(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self._event_queue.join()
        self._ev_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.flush()
        self._ev_writer.close()
        self._closed = True


class _EventLoggerThread(threading.Thread):
    """Thread that logs events."""

    def __init__(self, queue, ev_writer, flush_secs):
        """Creates an _EventLoggerThread.
        Args:
          queue: A Queue from which to dequeue events.
          ev_writer: An event writer. Used to log brain events for
           the visualizer.
          flush_secs: How often, in seconds, to flush the
            pending file to disk.
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0

    def run(self):
        while True:
            event = self._queue.get()
            try:
                self._ev_writer.write_event(event)
                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    self._ev_writer.flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
