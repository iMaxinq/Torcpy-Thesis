## Experimental Execution

Prior to execution, ensure that the input artifact `input_noisy.mp4` resides in the root execution directory alongside the source code.

### Hostfile Generation

Construct a configuration file named `hostfile` (or `machinefile`). This file must detail the IP addresses of the cluster nodes and the corresponding number of MPI slots (processes) allocated to each.

**Example `hostfile` configuration:**

```text
localhost slots=1
192.168.1.101 slots=1
192.168.1.102 slots=1
```

### Execution Command

To initiate the distributed computation, invoke the `mpiexec` (or `mpirun`) command, supplying the hostfile and specifying the aggregate number of processes (`-n`):

```bash
mpiexec -n 3 -f hostfile python3 main.py
```

> **Note:** Modify `main.py` to match the exact filename of your primary Python script.

Upon successful termination, the algorithmically denoised frames will be systematically archived in the `./output_frames` directory.

---

## Runtime Configuration and Task Scheduling (TorcPy)

The TorcPy runtime environment parameters are configured dynamically through **environment variables**. These must be exported prior to invoking the `mpiexec` command.

The available configuration parameters are summarized below.

| Variable | Description | Default | Permitted Values |
|----------|-------------|---------|------------------|
| `TORCPY_SCHEDULING` | Specifies the scheduling policy used for task distribution across nodes. | `round_robin` | `round_robin`, `weighted`, `heft` |
| `TORCPY_WORKERS` | Defines the number of worker threads per MPI process. | `1` | Any positive integer |
| `TORCPY_STEALING` | Enables distributed work-stealing for dynamic load balancing. | `False` | `True`, `False` |
| `TORCPY_PRINT_STATS` | Prints execution statistics and scheduling diagnostics after completion. | `False` | `True`, `False` |

### Example Configuration

The following example executes the application using the **HEFT** scheduler, **4 worker threads** per MPI process, and enables execution statistics.

```bash
export TORCPY_SCHEDULING="heft"
export TORCPY_WORKERS=4
export TORCPY_PRINT_STATS="True"

mpiexec -n 3 -f hostfile python3 main.py
```

---

## CPU Utilization and Threading

The implementation explicitly forces the underlying native libraries (OpenBLAS, MKL, and OpenCV) to operate in **single-threaded mode** through `os.environ` overrides.

This design choice prevents CPU oversubscription, ensuring that all application-level parallelism is managed exclusively by the TorcPy runtime and its scheduling policies rather than by the individual numerical libraries.
