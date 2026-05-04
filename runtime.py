"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""
from mpi4py.MPI import pickle

"""Implements torcpy runtime system and API."""

import copy
import ctypes
import queue
import threading
import time
import sys
import os
from termcolor import cprint
from functools import partial

# import mpi4py
# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False
import builtins
import itertools
import logging
import coloredlogs

_torc_log = logging.getLogger(__name__)
_torc_log.setLevel(logging.DEBUG)

coloredlogs.install(fmt='[%(name)s %(levelname)s]  %(message)s', stream=sys.stdout, level=logging.DEBUG)


class COMM(object):
    def __init__(self):
        self.rank = 0
        self.size = 1
    def Get_rank(self):
        return self.rank
    def Get_size(self):
        return self.size
    def barrier(self):
        pass


class myMPI(object):
    def __init__(self):
        self.COMM_WORLD = COMM()
        self.THREAD_MULTIPLE = '3'

    def Query_thread(self):
        return self.THREAD_MULTIPLE


try:
    from mpi4py import MPI
    _torc_log.debug('mpi4py module was succesfully imported')
except:
    _torc_log.warning("mpi4py could not be imported, loading a dummy MPI module.")
    MPI = myMPI()


# Constants - to be moved to constants.py
TORC_SERVER_TAG = 100
TORC_STEAL_RESPONSE_TAG = 101
TORC_HEARTBEAT_TAG = 102
TORC_LOAD_UPDATE_INTERVAL = 0.1
TORC_TASK_YIELDTIME = 0.0001
TORC_QUEUE_LEVELS = 10

# Environment variables
TORC_STEALING_ENABLED = False
TORC_SERVER_YIELDTIME = 0.01
TORC_WORKER_YIELDTIME = 0.01
TORC_SCHEDULING = "round_robin"
TORC_PRINT_STATS = False

# TORC data: task queue, thread local storage, MPI communicator
torc_q = []
for _ in range(TORC_QUEUE_LEVELS+1):
    torc_q.append(queue.Queue())
torc_tls = threading.local()
torc_comm = MPI.COMM_WORLD
torc_server_thread = None
torc_deps_lock = threading.Lock()
torc_use_server = True

torc_last_qid = -1  # will need an extra lock for this
torc_executed = 0
torc_created = 0
torc_stats_lock = threading.Lock()
torc_steal_lock = threading.Lock()
torc_num_workers = int(os.getenv("TORCPY_WORKERS", 1))
torc_workers = []
torc_exit_flag = False

# Torc task history
torc_task_history = []
torc_history_lock = threading.Lock()
torc_total_steal_attempts = 0
torc_total_stolen_bytes = 0
torc_steal_stats_lock = threading.Lock()

# Torc weighted scheduling variables
torc_node_benchmark_times = []
torc_node_weights = []
torc_weighted_rr_state = []
torc_weighted_node_index = 0
torc_sched_lock = threading.Lock()

# Torc HEFT scheduling variables
torc_hostnames = []
torc_intra_rate_bps = float('inf')  # Bytes per second (Same Chip)
torc_inter_rate_bps = float('inf')  # Bytes per second (Wi-Fi)
TORC_ESTIMATED_TASK_BYTES = 0
torc_node_estimated_load = []

# Flags
_torc_shutdowned = False
_torc_launched = False
_torc_inited = False


def node_id():
    # Rank of calling MPI process
    return torc_comm.Get_rank()


def num_nodes():
    # Number of MPI processes
    return torc_comm.Get_size()


def worker_local_id():
    # Local id of current worker thread
    return torc_tls.id


def num_local_workers():
    # Number of local workers
    return torc_num_workers


def worker_id():
    # Global id of current worker thread
    return node_id() * torc_num_workers + torc_tls.id


def num_workers():
    # Total number of workers
    return num_nodes() * torc_num_workers


def _send_desc_and_data(qid, task):
    torc_comm.send(task, dest=qid, tag=TORC_SERVER_TAG)


class TaskT:
    def __init__(self, desc):
        self.desc = desc

    def input(self):
        return self.desc["args"]

    def kw_input(self):
        return self.desc["kwargs"]

    def result(self):
        return self.desc["out"]

    def __del__(self):
        if "parent" in self.desc:
            del self.desc["parent"]
        if "args" in self.desc:
            del self.desc["args"]
        if "kwargs" in self.desc:
            del self.desc["kwargs"]
        del self.desc

def _schedule_round_robin():
    """Return the next worker id using round-robin scheduling."""
    global torc_last_qid

    with torc_stats_lock:
        qid = torc_last_qid
        torc_last_qid = qid + 1
        if torc_last_qid == num_workers():
            torc_last_qid = 0

    return qid

def _schedule_weighted():
    """Schedule according to node benchmark weights."""
    global torc_weighted_node_index

    with torc_sched_lock:
        if not torc_node_weights or not torc_weighted_rr_state:
            return _schedule_round_robin()

        target_node = torc_weighted_rr_state[torc_weighted_node_index]
        torc_weighted_node_index = (torc_weighted_node_index + 1) % len(torc_weighted_rr_state)

        qid = target_node * num_local_workers()

    return qid

def _schedule_HEFT(args, kwargs):
    """Return the next worker id using Dynamic Earliest Finish Time."""
    global torc_node_benchmark_times
    global torc_hostnames, torc_intra_rate_bps, torc_inter_rate_bps
    global TORC_ESTIMATED_TASK_BYTES

    if not torc_node_benchmark_times or not torc_hostnames:
        return _schedule_round_robin()

    best_node = 0
    min_eft = float('inf')
    my_hostname = MPI.Get_processor_name()
    param_bytes = _estimate_payload_bytes(args, kwargs)
    total_task_bytes = TORC_ESTIMATED_TASK_BYTES + param_bytes

    with torc_sched_lock:
        for i in range(num_nodes()):
            # Computational Cost
            exec_cost = torc_node_benchmark_times[i]

            # Communication Cost
            if i == node_id():
                comm_cost = 0.0  # Same process
            elif torc_hostnames[i] == my_hostname:
                comm_cost = total_task_bytes / torc_intra_rate_bps  # Same chip
            else:
                comm_cost = total_task_bytes / torc_inter_rate_bps  # Via wifi

            # Load Cost
            current_tasks = torc_node_estimated_load[i]
            ready_time = (current_tasks / num_local_workers()) * exec_cost

            # Calculate Earliest Finish Time
            eft = ready_time + comm_cost + exec_cost

            if eft < min_eft:
                min_eft = eft
                best_node = i

        print(best_node)
        qid = best_node * num_local_workers()

    return qid

def _estimate_payload_bytes(args, kwargs):
    """Fast estimation of argument byte size to avoid pickling overhead."""
    total_bytes = 0

    # Check positional args
    for arg in args:
        if hasattr(arg, 'nbytes'):  # Instant accurate size for NumPy arrays
            total_bytes += arg.nbytes
        else:
            total_bytes += sys.getsizeof(arg)  # Fast but shallow estimate

    # Check keyword args
    for k, v in kwargs.items():
        total_bytes += sys.getsizeof(k)
        if hasattr(v, 'nbytes'):
            total_bytes += v.nbytes
        else:
            total_bytes += sys.getsizeof(v)

    return total_bytes

def submit(f, *a, qid=-1, callback=None, async_callback=True, counted=True, **kwargs):
    """Submit a task to be executed with the given arguments.

    Args:
        f: A callable to be executed as ``f(*a)``
        a: input arguments
        qid: target task queue
        callback: function to be executed on the results of ``f(*a)``
        async_callback: if True, the callback is inserted in the queue, otherwise executed immediately
        counted: if False, the task and its callback are not added to the counter of created tasks

    Returns:
        A `Future` representing the given call.
    """

    global torc_last_qid
    global torc_created

    if qid is not None and qid >= num_workers():
        _torc_log.error("submit: invalid qid value ({})".format(qid))
        raise ValueError

    # Cyclic task distribution
    # Scheduling policy
    if qid == -1:
        if TORC_SCHEDULING == "round_robin":
            qid = _schedule_round_robin()
        elif TORC_SCHEDULING == "weighted":
            qid = _schedule_weighted()
        elif TORC_SCHEDULING == "heft":
            qid = _schedule_HEFT(a, kwargs)

    if qid is not None:
        qid = qid % num_workers()
        qid = int(qid / num_local_workers())

    # Prepare the task descriptor
    task = dict()
    task["t_ready"] = time.time()
    task["varg"] = True
    task["mytask"] = id(task)
    task["f"] = f
    task["cb"] = callback
    task["async_callback"] = async_callback
    if len(a) == 1:
        task["varg"] = False
        task["args"] = copy.copy(*a)  # firstprivate
    else:
        task["args"] = copy.copy(a)  # firstprivate
    task["kwargs"] = copy.copy(kwargs)

    # _torc_log.info("kwargs={}".format(task["kwargs"]))
    task["out"] = None  #
    task["homenode"] = node_id()
    task["deps"] = 0
    task["counted"] = counted
    task["completed"] = []
    parent = torc_tls.curr_task
    task_level = parent["level"] + 1
    if task_level >= TORC_QUEUE_LEVELS:
        task_level = TORC_QUEUE_LEVELS - 1
    task["level"] = task_level

    cb_task = None
    if callback is not None:
        cb_task = dict()
        cb_task["t_ready"] = time.time()
        cb_task["varg"] = False
        cb_task["mytask"] = id(cb_task)
        cb_task["f"] = callback
        cb_task["kwargs"] = {}
        cb_task["args"] = None
        cb_task["out"] = None  #
        cb_task["homenode"] = node_id()
        cb_task["deps"] = 0
        cb_task["counted"] = counted
        cb_task["cbtask"] = None
        cb_task["level"] = task_level

    if qid is not None:
        parent = torc_tls.curr_task
        task["parent"] = id(parent)
        torc_deps_lock.acquire()
        parent["deps"] += 1
        torc_deps_lock.release()

        if counted:
            torc_stats_lock.acquire()
            torc_created += 1
            torc_stats_lock.release()

        if callback is not None:
            cb_task["parent"] = id(parent)

            if counted:
                torc_stats_lock.acquire()
                torc_created += 1
                torc_stats_lock.release()

            if async_callback:
                torc_deps_lock.acquire()
                parent["deps"] += 1
                torc_deps_lock.release()
            task["cbtask"] = cb_task
        else:
            task["cbtask"] = None

    else:
        return

    # Enqueue to local or remote queue
    if node_id() == qid:
        enqueue(task_level, task)
    else:
        task["type"] = "enqueue"
        _send_desc_and_data(qid, task)

    # dictionary to object
    task_obj = TaskT(task)
    return task_obj


def _do_work(task):
    global torc_executed

    if num_nodes() > 1:
        time.sleep(TORC_TASK_YIELDTIME)

    torc_tls.curr_task = task

    f = task["f"]
    args = task["args"]
    kwargs = task["kwargs"]
    task["t_start"] = time.time()

    if task["varg"]:
        y = f(*args, **kwargs)
    else:
        if args is None:  # to be safe
            y = f()
        else:
            y = f(args, **kwargs)

    task["t_finish"] = time.time()

    # create logs
    if task.get("counted", True):
        q_len = sum(torc_q[i].qsize() for i in range(TORC_QUEUE_LEVELS))
        with torc_history_lock:
            torc_task_history.append({
                "task_id": task["mytask"],
                "worker": worker_id(),
                "t_ready": task.get("t_ready", 0.0),
                "t_start": task["t_start"],
                "t_finish": task["t_finish"],
                "q_len": q_len
            })

    # send answer and results back to the homenode of the task
    if node_id() == task["homenode"]:

        task = ctypes.cast(task["mytask"], ctypes.py_object).value  # real task
        task["out"] = copy.copy(y)

        # satisfy dependencies
        parent = ctypes.cast(task["parent"], ctypes.py_object).value
        torc_deps_lock.acquire()
        parent["deps"] -= 1
        parent["completed"].append(task)
        torc_deps_lock.release()

        # trigger the callback function
        cb_task = task["cbtask"]
        if cb_task is not None:
            cb_task["args"] = TaskT(task)
            if task["async_callback"]:
                enqueue(cb_task["level"], cb_task)
            else:
                cb_task["f"](cb_task["args"])
                if task["counted"]:
                    torc_stats_lock.acquire()
                    torc_executed += 1
                    torc_stats_lock.release()

    else:
        task["out"] = y
        dest = task["homenode"]

        task["type"] = "answer"
        del task["args"]
        del task["kwargs"]
        # avoid redundant byte transfers (hint: del more)
        torc_comm.send(task, dest=dest, tag=TORC_SERVER_TAG)

    if task["counted"]:
        torc_stats_lock.acquire()
        torc_executed += 1
        torc_stats_lock.release()


def enqueue(level, task):
    """ Insert task to the queue
    """
    torc_q[level].put(task)


def dequeue():
    """Extract a task from a queue
    """
    for i in range(TORC_QUEUE_LEVELS):
        try:
            task = torc_q[i].get(True, 0)
            return task
        except queue.Empty:
            continue

    # check also the special queue for termination signal
    try:
        task = torc_q[TORC_QUEUE_LEVELS].get(True, 0)
    except queue.Empty:
        time.sleep(TORC_WORKER_YIELDTIME)
        task = {}

    return task


def dequeue_steal():
    """Extract (steal) a task from a queue
    """
    for i in range(TORC_QUEUE_LEVELS-1, -1, -1):
        try:
            task = torc_q[i].get(False)
            return task
        except queue.Empty:
            continue

    task = {}
    return task


def waitall(tasks=None, as_completed=False):
    """Suspend the calling task until all each spawned child tasks have completed
    """
    global torc_executed
    mytask = torc_tls.curr_task  # myself

    completed_tasks = None
    while True:
        torc_deps_lock.acquire()
        deps = mytask["deps"]
        torc_deps_lock.release()

        if deps == 0:
            if as_completed:
                completed_tasks = copy.copy(mytask["completed"])
            mytask["completed"] = []
            break

        while True:
            task = dequeue()
            if task is None:
                break

            if not task:
                if num_nodes() > 1 and TORC_STEALING_ENABLED:
                    task = _steal()
                    if task["type"] == "nowork":
                        time.sleep(TORC_WORKER_YIELDTIME)
                        break
                    else:
                        pass
                else:
                    break

            _do_work(task)

    torc_tls.curr_task = mytask
    return completed_tasks


def wait(tasks=None):
    # For PEP 3148
    if tasks is None:
        waitall(tasks)
    else:
        waitall(tasks)
    return tasks


def as_completed(tasks=None):
    completed_tasks = waitall(tasks, as_completed=True)
    t = []
    for c in completed_tasks:
        t.append(TaskT(c))
    return t


def _worker(w_id):
    torc_tls.id = w_id

    while True:
        while True:
            task = dequeue()
            if task is None:
                break

            if not task:
                if num_nodes() > 1 and TORC_STEALING_ENABLED:
                    task = _steal()
                    if task["type"] == "nowork":
                        time.sleep(TORC_WORKER_YIELDTIME)
                        break
                    else:
                        pass
                else:
                    break

            _do_work(task)

        if task is None:
            break


def _server():
    global torc_exit_flag, torc_executed
    global torc_node_estimated_load

    torc_tls.id = -1

    status = MPI.Status()  # get MPI status object

    last_load_update = time.time()
    active_reqs = []

    while True:
        has_server_msg = torc_comm.Iprobe(source=MPI.ANY_SOURCE, tag=TORC_SERVER_TAG, status=status)
        if has_server_msg:
            source_rank = status.Get_source()
            source_tag = status.Get_tag()
            task = torc_comm.recv(source=source_rank, tag=source_tag, status=status)

            ttype = task["type"]

            if ttype == "exit":
                torc_exit_flag = True
                for i in range(torc_num_workers):
                    enqueue(TORC_QUEUE_LEVELS, None)
                break

            elif ttype == "enqueue":
                task["source_rank"] = source_rank
                task["source_tag"] = source_tag
                enqueue(task["level"], task)

            elif ttype == "answer":
                real_task = ctypes.cast(task["mytask"], ctypes.py_object).value
                real_task["out"] = copy.copy(task["out"])

                parent = ctypes.cast(task["parent"], ctypes.py_object).value
                torc_deps_lock.acquire()
                parent["deps"] -= 1
                parent["completed"].append(real_task)
                torc_deps_lock.release()

                cb_task = real_task["cbtask"]
                if cb_task is not None:
                    cb_task["args"] = TaskT(real_task)
                    if real_task["async_callback"]:
                        enqueue(cb_task["level"], cb_task)
                    else:
                        cb_task["f"](cb_task["args"])
                        if real_task["counted"]:
                            torc_stats_lock.acquire()
                            torc_executed += 1
                            torc_stats_lock.release()

            elif ttype == "steal":
                t = dequeue_steal()
                if t is None:
                    torc_q[TORC_QUEUE_LEVELS].put(t)
                    t = dict()
                    t["type"] = "nowork"
                elif not t:
                    t["type"] = "nowork"
                else:
                    t["type"] = "stolen"
                torc_comm.send(t, dest=source_rank, tag=TORC_STEAL_RESPONSE_TAG)

        has_load_update_msg = torc_comm.Iprobe(source=MPI.ANY_SOURCE, tag=TORC_HEARTBEAT_TAG, status=status)
        if has_load_update_msg:
            source_rank = status.Get_source()
            q_size = torc_comm.recv(source=source_rank, tag=TORC_HEARTBEAT_TAG, status=status)
            torc_node_estimated_load[source_rank] = q_size

        if not has_server_msg and not has_load_update_msg:
            time.sleep(TORC_SERVER_YIELDTIME)

        now = time.time()

        if now - last_load_update >= TORC_LOAD_UPDATE_INTERVAL:
            local_q_size = sum(torc_q[i].qsize() for i in range(TORC_QUEUE_LEVELS))
            torc_node_estimated_load[node_id()] = local_q_size

            for i in range(num_nodes()):
                if i != node_id():
                    req = torc_comm.isend(local_q_size, dest=i, tag=TORC_HEARTBEAT_TAG)
                    active_reqs.append(req)

            last_load_update = now

        active_reqs = [req for req in active_reqs if not req.Test()]


"""
Notes on the work stealing implementation:
- It is synchronous (direct response and execution of the task)
  - same tag, but it should work for multiple threads even without explicit synchronization
- Asynchronous version 1: the worker puts the stolen task in the queue
- Asynchronous version 2: the stolen task is sent to the server thread
- Better visit of remote nodes, bookkeeping of last node that had available tasks
"""


def _steal():
    global TORC_STEALING_ENABLED, torc_exit_flag
    global torc_total_steal_attempts, torc_total_stolen_bytes

    task = None
    status = MPI.Status()  # get MPI status object
    task_req = dict()
    task_req["type"] = "steal"
    me = node_id()
    n = num_nodes()
    for i in range(0, n):
        if i == me:
            continue
        if torc_exit_flag or not TORC_STEALING_ENABLED:
            task = dict()
            task["type"] = "nowork"
            break

        with torc_steal_stats_lock:
            torc_total_steal_attempts += 1

        # torc_steal_lock.acquire()
        torc_comm.send(task_req, dest=i, tag=TORC_SERVER_TAG)
        # OLD code, can lead to deadlock during shutdown
        # task = torc_comm.recv(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status)

        while not torc_comm.Iprobe(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status):
            time.sleep(TORC_WORKER_YIELDTIME)
            if torc_exit_flag:
                break

        if not torc_exit_flag:
            task = torc_comm.recv(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status)
            if task["type"] != "nowork":
                stolen_bytes = len(pickle.dumps(task))
                with torc_steal_stats_lock:
                    torc_total_stolen_bytes += stolen_bytes

        else:
            task = dict()
            task["type"] = "nowork"
            # torc_steal_lock.release()
            break

        # torc_steal_lock.release()
        if task["type"] == "nowork":
            continue
        else:
            task["source_rank"] = i
            task["source_tag"] = TORC_STEAL_RESPONSE_TAG  # not used
            break

    return task


def init():
    """Initializes the runtime library """
    global torc_server_thread, torc_use_server
    global torc_last_qid
    global torc_num_workers
    global TORC_STEALING_ENABLED
    global TORC_SERVER_YIELDTIME, TORC_WORKER_YIELDTIME, TORC_SCHEDULING, TORC_PRINT_STATS
    global _torc_inited

    if _torc_inited is True:
        return
    else:
        _torc_inited = True

    # MPI.Init_thread()

    provided = MPI.Query_thread()
    if MPI.COMM_WORLD.Get_rank() == 0:
        _torc_log.warning("MPI.Query_thread returns {}".format(provided))
        if provided < MPI.THREAD_MULTIPLE:
            _torc_log.warning("Warning: MPI.Query_thread returns {} < {}".format(provided, MPI.THREAD_MULTIPLE))
        else:
            _torc_log.warning("Info: MPI.Query_thread returns MPI.THREAD_MULTIPLE")

    torc_num_workers = int(os.getenv("TORCPY_WORKERS", 1))

    flag = os.getenv("TORCPY_STEALING", "False")
    if flag == "True":
        TORC_STEALING_ENABLED = True
    else:
        TORC_STEALING_ENABLED = False

    flag_stats = os.getenv("TORCPY_PRINT_STATS", "False").strip().lower()
    TORC_PRINT_STATS = (flag_stats == "true")

    TORC_SERVER_YIELDTIME = float(os.getenv("TORCPY_SERVER_YIELDTIME", 0.01))
    TORC_WORKER_YIELDTIME = float(os.getenv("TORCPY_WORKER_YIELDTIME", 0.01))
    TORC_SCHEDULING = os.getenv("TORCPY_SCHEDULING", "round_robin").strip().lower()

    if TORC_SCHEDULING not in ("round_robin", "weighted", "heft"):
        raise ValueError("Invalid TORCPY_SCHEDULING='{}'; expected 'round_robin' or 'weighted' or 'heft'".format(TORC_SCHEDULING))

    torc_tls.id = 0
    main_task = dict()
    main_task["deps"] = 0
    main_task["mytask"] = id(main_task)
    main_task["parent"] = 0
    main_task["level"] = -1
    main_task["completed"] = []
    torc_tls.curr_task = main_task
    torc_last_qid = node_id() * torc_num_workers

    # initialize load array
    global torc_node_estimated_load
    torc_node_estimated_load = [0] * num_nodes()

    if num_nodes() == 1:
        torc_use_server = False

    if torc_use_server:
        torc_server_thread = threading.Thread(target=_server)
        torc_server_thread.start()


def _terminate_nodes():
    task = dict()
    task["f"] = 0
    task["type"] = "exit"

    # disable_stealing()

    nodes = num_nodes()
    for i in range(0, nodes):
        torc_comm.send(task, dest=i, tag=TORC_SERVER_TAG)


def _print_stats():
    global torc_created, torc_executed, torc_total_steal_attempts, torc_total_stolen_bytes
    me = node_id()
    msg = "\nTORCPY: node[{}]: created={}, executed={}".format(me, torc_created, torc_executed)
    steal_msg = f"        node[{me}]: total steal attempts={torc_total_steal_attempts}, stolen bytes={torc_total_stolen_bytes} bytes"
    cprint(msg, "green")
    cprint(steal_msg, "cyan")

    print(f"=== Node {me} Task History ===")
    for stat in torc_task_history:
        ready_to_start = stat['t_start'] - stat['t_ready']
        duration = stat['t_finish'] - stat['t_start']
        print(f"Worker {stat['worker']} | Task {stat['task_id']} | "
              f"Wait: {ready_to_start:.4f}s | Run: {duration:.4f}s | "
              f"Queue Len: {stat['q_len']}")

    sys.stdout.flush()  # Force output to terminal before the next node prints
    torc_comm.barrier()  # Wait for the current node to finish printing

    torc_created = 0
    torc_executed = 0
    torc_total_steal_attempts = 0
    torc_total_stolen_bytes = 0


def finalize():
    """Shutdowns the runtime library
    """
    global torc_server_thread, torc_workers

    if node_id() == 0:

        for _ in torc_workers:
            torc_q[0].put(None)

        for t in torc_workers:
            t.join()

        if torc_use_server:
            _terminate_nodes()

    else:
        pass

    if torc_use_server:
        torc_server_thread.join()

    if TORC_PRINT_STATS:
        _print_stats()
    torc_comm.barrier()


def shutdown():
    # For PEP 3148
    global _torc_shutdowned
    if _torc_shutdowned is True:
        return
    else:
        _torc_shutdowned = True
    finalize()


def launch(main_function):
    """Launches `main_function` on worker 0 of rank 0 as the primary task of the application
    """
    global torc_workers
    global _torc_launched

    if _torc_launched is True:
        return
    else:
        _torc_launched = True

    if node_id() == 0:
        cprint("TORCPY: main starts", "green")
        sys.stdout.flush()
        torc_comm.barrier()

        torc_workers = []
        for i in range(1, torc_num_workers):
            torc_worker_thread = threading.Thread(target=_worker, args=(i,))
            torc_worker_thread.start()
            torc_workers.append(torc_worker_thread)

        torc_comm.barrier()
        if main_function is not None:
            main_function()

    else:
        torc_comm.barrier()

        torc_workers = []
        for i in range(1, torc_num_workers):
            torc_worker_thread = threading.Thread(target=_worker, args=(i,))
            torc_worker_thread.start()
            torc_workers.append(torc_worker_thread)

        torc_comm.barrier()
        _worker(0)

        for t in torc_workers:
            t.join()

        sys.stdout.flush()
        if main_function is None:
            finalize()
            sys.exit(0)

# weighted scheduling functions
def _run_node_benchmark(bench_f, *args, **kwargs):
    t0 = time.time()
    bench_f(*args, **kwargs)
    t1 = time.time()
    return t1 - t0

def _compute_node_weights(times):
    eps = 1e-12
    speeds = []

    for t in times:
        if t is None or t <= 0:
            speeds.append(eps)
        else:
            speeds.append(1.0 / t)

    total_speed = sum(speeds)
    if total_speed <= 0:
        n = len(times)
        return [1.0 / n for _ in range(n)]

    return [s / total_speed for s in speeds]

def _build_weighted_rr_state(weights, table_size=100):
    counts = [max(1, int(round(w * table_size))) for w in weights]
    total_slots = sum(counts)

    positions = []
    for node, count in enumerate(counts):
        for i in range(count):
            ideal_idx = (i + 0.5) * (total_slots / count)
            positions.append((ideal_idx, node))

    positions.sort()
    state = [node for ideal_idx, node in positions]

    if not state:
        state = list(range(num_nodes()))

    return state

def init_node_weights(bench_f, *args, **kwargs):
    global torc_node_benchmark_times
    global torc_node_weights
    global torc_weighted_rr_state
    global torc_weighted_node_index
    global TORC_STEALING_ENABLED

    # Disable stealing so that no node steals the benchmark task
    original_stealing_state = TORC_STEALING_ENABLED
    TORC_STEALING_ENABLED = False

    t_all = []
    for node in range(num_nodes()):
        task = submit(
            _run_node_benchmark,
            bench_f,
            *args,
            qid=node * num_local_workers(),
            counted=False,
            **kwargs
        )
        t_all.append(task)

    waitall()

    # restore stealing state
    TORC_STEALING_ENABLED = original_stealing_state

    times = [task.result() for task in t_all]
    weights = _compute_node_weights(times)
    state = _build_weighted_rr_state(weights)

    with torc_sched_lock:
        torc_node_benchmark_times = times
        torc_node_weights = weights
        torc_weighted_rr_state = state
        torc_weighted_node_index = 0

    if node_id() == 0:
        _torc_log.warning("Benchmark times per node: {}".format(times))
        _torc_log.warning("Derived node weights: {}".format(weights))
        _torc_log.warning("State table: {}".format(state))

# hardcoded benchmarks provided by the library
def benchmark_cpu():
    """Stresses the ALU with tight integer math loops."""
    x = 0
    for i in range(5_000_000):
        x += (i % 97) * (i % 89)
    return x

def benchmark_memory():
    """Stresses RAM bandwidth and cache by allocating and modifying a large array."""
    size = 5_000_000
    data = [0] * size
    for i in range(size):
        data[i] = i % 256
    return sum(data)

def benchmark_io():
    """Simulates I/O bound tasks (like network requests or disk reads). Essentially, round-robin"""
    time.sleep(0.5)
    return True

# HEFT scheduling functions
def _benchmark_network_rates(payload_bytes=1024 * 1024, iterations=10):
    global torc_hostnames, torc_intra_rate_bps, torc_inter_rate_bps

    my_hostname = MPI.Get_processor_name()
    all_hostnames = torc_comm.allgather(my_hostname)  # all processor names sorted by rank

    me = node_id()
    n_nodes = num_nodes()

    local_target = None
    remote_target = None

    # Rank 0 figures out who to test with
    if me == 0:
        for i in range(1, n_nodes):
            if all_hostnames[i] == my_hostname and local_target is None:
                local_target = i
            elif all_hostnames[i] != my_hostname and remote_target is None:
                remote_target = i

    # Broadcast the chosen targets so those specific ranks know to participate
    local_target, remote_target = torc_comm.bcast((local_target, remote_target), root=0)

    dummy_task = bytearray(payload_bytes)

    # Intra-node benchmark (Local)
    intra_rate = float('inf')
    if local_target is not None:
        if me == 0:
            torc_comm.barrier()
            times = []
            for _ in range(iterations):
                t0 = time.time()
                torc_comm.send(dummy_task, dest=local_target, tag=999)
                torc_comm.recv(source=local_target, tag=999)
                t1 = time.time()
                times.append(t1 - t0)

            avg_ping_pong = sum(times) / len(times)
            avg_send_time = avg_ping_pong / 2.0

            # bytes/sec rate
            intra_rate = payload_bytes / avg_send_time

        elif me == local_target:
            torc_comm.barrier()
            for _ in range(iterations):
                msg = torc_comm.recv(source=0, tag=999)
                torc_comm.send(msg, dest=0, tag=999)
        else:
            torc_comm.barrier()

    # Inter-node benchmark (Wi-Fi/Network)
    inter_rate = float('inf')
    if remote_target is not None:
        if me == 0:
            torc_comm.barrier()
            times = []
            for _ in range(iterations):
                t0 = time.time()
                torc_comm.send(dummy_task, dest=remote_target, tag=999)
                torc_comm.recv(source=remote_target, tag=999)
                t1 = time.time()
                times.append(t1 - t0)

            avg_ping_pong = sum(times) / len(times)
            avg_send_time = avg_ping_pong / 2.0

            # bytes/sec rate
            inter_rate = payload_bytes / avg_send_time

        elif me == remote_target:
            torc_comm.barrier()
            for _ in range(iterations):
                msg = torc_comm.recv(source=0, tag=999)
                torc_comm.send(msg, dest=0, tag=999)
        else:
            torc_comm.barrier()

    intra_rate = torc_comm.bcast(intra_rate, root=0)
    inter_rate = torc_comm.bcast(inter_rate, root=0)

    with torc_sched_lock:  # not needed now but could be later
        torc_hostnames = all_hostnames
        torc_intra_rate_bps = intra_rate
        torc_inter_rate_bps = inter_rate

    if me == 0:
        _torc_log.info(
            f"Network Bench: Intra-node = {intra_rate / 1e6:.2f} MB/s | Inter-node = {inter_rate / 1e6:.2f} MB/s")

def _calculate_base_task_bytes():
    global TORC_ESTIMATED_TASK_BYTES

    dummy_task = dict()
    dummy_task["t_ready"] = time.time()
    dummy_task["varg"] = False
    dummy_task["mytask"] = id(dummy_task)
    dummy_task["f"] = None
    dummy_task["cb"] = None
    dummy_task["async_callback"] = False
    dummy_task["args"] = None
    dummy_task["kwargs"] = {}
    dummy_task["out"] = None
    dummy_task["homenode"] = node_id()
    dummy_task["deps"] = 0
    dummy_task["counted"] = True
    dummy_task["completed"] = []
    dummy_task["level"] = 0
    dummy_task["cbtask"] = None
    dummy_task["parent"] = id(dummy_task)

    exact_bytes = len(pickle.dumps(dummy_task))

    if node_id() == 0:
        _torc_log.info(f"Calculated Base Task Overhead: {exact_bytes} bytes")

    TORC_ESTIMATED_TASK_BYTES = exact_bytes

def start(main_function, profile="cpu"):
    """Initialize the library, start the primary application task, and shutdown."""

    init()

    if TORC_SCHEDULING in ("weighted", "heft"):
        # Determine which benchmark to run
        if callable(profile):
            bench_f = profile
            _torc_log.info("Using custom user-defined benchmark profile.")
        elif profile == "cpu":
            bench_f = benchmark_cpu
            _torc_log.info("Using built-in 'CPU' benchmark profile.")
        elif profile == "memory":
            bench_f = benchmark_memory
            _torc_log.info("Using built-in 'MEMORY' benchmark profile.")
        elif profile == "io":
            bench_f = benchmark_io
            _torc_log.info("Using built-in 'IO' benchmark profile.")
        else:
            _torc_log.warning(f"Unknown profile '{profile}'. Defaulting to 'cpu'.")
            bench_f = benchmark_cpu

        init_node_weights(bench_f)

        if TORC_SCHEDULING == "heft":
            _calculate_base_task_bytes()
            _benchmark_network_rates()

    launch(main_function)
    finalize()


def gettime():
    """Returns current time in seconds. For compatibility with the C version of the library
    """
    return time.time()


def spmd(spmd_task, *arg, counted=True):
    """Submit a task to be executed with the given arguments on all MPI processes.
       Wait for completion.

    Args:
        spmd_task: A callable to be executed as ``spmd_task(*args)``
        arg: input arguments
        counted: if True, the spawned task are included in the statistics

    Returns:
        Nothing
    """

    t_all = []
    ntasks = num_nodes()
    for i in range(0, ntasks):
        task = submit(spmd_task, *arg, qid=i*num_local_workers(), counted=counted)
        t_all.append(task)
    waitall()


def _enable_stealing_task():
    global TORC_STEALING_ENABLED

    TORC_STEALING_ENABLED = True


def _disable_stealing_task():
    global TORC_STEALING_ENABLED

    TORC_STEALING_ENABLED = False


def enable_stealing():
    # Enable stealing
    spmd(_enable_stealing_task, counted=False)


def disable_stealing():
    # Disable stealing
    spmd(_disable_stealing_task, counted=False)


# map with chunksize
def _apply_chunks(f, chunk):
    return [f(*args) for args in chunk]


def _build_chunks(chunksize, iterable):
    iterable = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterable, chunksize))
        if not chunk:
            return
        yield (chunk)


def map(f, *seq, chunksize=1):
    """Return an iterator equivalent to ``map(f, *seq)``.
    Args:
        f: A callable that will take as many arguments as there are
            passed iterables.
        seq: Iterables yielding positional arguments to be passed to
                the callable.
        chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.

    Returns:
        An iterator equivalent to built-in ``map(func, *iterables)``
        but the calls may be evaluated out-of-order.

    Raises:
        Exception: If ``f(*args)`` raises for any values.
    """

    if chunksize == 1:
        t_all = list(builtins.map(partial(submit, f), *seq))
        waitall()

        res = []
        for task in t_all:
            res.append(task.result())

        return res

    else:
        iterable = getattr(itertools, 'izip', zip)(*seq)
        new_seq = list(_build_chunks(chunksize, iterable))
        f1 = partial(_apply_chunks, f)
        t_all = list(builtins.map(partial(submit, f1), new_seq))
        waitall()

        res = []
        for task in t_all:
            res.append(task.result())

        flat_res = [item for sublist in res for item in sublist]
        return flat_res


torc_submit = submit
torc_map = map
torc_wait = wait


class TorcPoolExecutor:
    def __init__(self):
        launch(None)

    def __enter__(self):
        return self

    @staticmethod
    def submit(f, *a, qid=-1, callback=None, async_callback=True, counted=True):
        return torc_submit(f, *a, qid=qid, callback=callback, async_callback=async_callback, counted=counted)

    @staticmethod
    def map(f, *seq, chunksize=1):
        return torc_map(f, seq, chunksize=chunksize)

    @staticmethod
    def wait(tasks=None):
        return torc_wait(tasks=tasks)

    @staticmethod
    def shutdown():
        return torc_wait(tasks=None)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return torc_wait(tasks=None)

