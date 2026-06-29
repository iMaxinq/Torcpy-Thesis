# Distributed Video Denoising using TorcPy and MPI

This repository contains the software implementation developed as part of my thesis. It presents a distributed video processing framework, specifically targeting video denoising, utilizing the **Message Passing Interface (MPI)** via the **mpi4py** binding, alongside a custom task-scheduling runtime system designated as **TorcPy**.

The system is designed to ingest a video file, extract its constituent frames, distribute the workload across multiple nodes of a computational cluster for parallel processing (leveraging OpenCV for the denoising algorithms), and ultimately aggregate the processed frames into a synthesized output.

---

# Prerequisites

To deploy and execute this framework within a distributed cluster environment, the following dependencies are required:

* A **Linux-based Operating System** (e.g., Ubuntu/Debian distributions) on all constituent nodes.
* **Python 3.7** or higher.
* A functional installation of **MPICH** or **OpenMPI**.

---

# System Setup and Network Configuration

To facilitate seamless inter-node communication and process spawning by the MPI daemon without manual intervention, establishing **passwordless SSH authentication** across all machines within the cluster is strictly required.

## 1. Passwordless SSH Configuration

On the primary execution node (Master Node), generate an SSH key pair:

```bash
ssh-keygen -t rsa -b 4096
```

Press **Enter** at all prompts to create the key pair without a passphrase.

Next, copy the generated public key to every participating node in the cluster (including the master node itself if it also executes worker processes):

```bash
ssh-copy-id user@<NODE_1_IP>
ssh-copy-id user@<NODE_2_IP>
# Repeat for all cluster nodes...
```

Verify the configuration by connecting to each node:

```bash
ssh user@<NODE_IP>
```

If no password is requested, passwordless authentication has been configured successfully.

---

## 2. System-Level Dependencies

Install MPICH together with the required Python development packages on **every node**:

```bash
sudo apt-get update
sudo apt-get install -y mpich libmpich-dev python3 python3-pip python3-venv libgl1-mesa-glx
```

> **Note:** `libgl1-mesa-glx` is required to ensure proper execution of OpenCV in headless Linux environments.

---

## 3. Python Environment Initialization

It is recommended to isolate the project dependencies using a Python virtual environment.

Navigate to the repository directory and create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install mpi4py opencv-python termcolor coloredlogs
```

---

# Experimental Execution

Before execution, ensure that the input video (`input_noisy.mp4`) is located in the project's root directory.

## Hostfile Generation

Create a file named `hostfile` (or `machinefile`) describing the available cluster nodes and the number of MPI slots assigned to each.

Example:

```text
192.168.1.101:1
192.168.1.102:2
```

---

## Execution Command

Launch the distributed application using `mpiexec` (or `mpirun`):

```bash
mpiexec -n 3 -f hostfile python3 main.py
```

Replace `main.py` with the actual name of your application's entry-point script if different.

After successful completion, the denoised frames will be stored in:

```text
./output_frames
```

---

# Runtime Configuration (TorcPy)

The TorcPy runtime is configured through **environment variables**. These variables must be exported before launching the MPI application.

| Variable             | Description                                 | Default       | Allowed Values                    |
| -------------------- | ------------------------------------------- | ------------- | --------------------------------- |
| `TORCPY_SCHEDULING`  | Task scheduling algorithm.                  | `round_robin` | `round_robin`, `weighted`, `heft` |
| `TORCPY_WORKERS`     | Number of worker threads per MPI process.   | `1`           | Positive integer                  |
| `TORCPY_STEALING`    | Enables distributed work stealing.          | `False`       | `True`, `False`                   |
| `TORCPY_PRINT_STATS` | Prints execution and scheduling statistics. | `False`       | `True`, `False`                   |

## Example Configuration

The following example executes the application using the **HEFT** scheduler, four worker threads per MPI process, and enables runtime statistics.

```bash
export TORCPY_SCHEDULING="heft"
export TORCPY_WORKERS=4
export TORCPY_PRINT_STATS="True"

mpiexec -n 3 -f hostfile python3 main.py
```

---

# CPU Utilization and Threading

The implementation explicitly forces the underlying native libraries (OpenBLAS, MKL, and OpenCV) to operate in **single-threaded mode** using `os.environ` overrides.

This design prevents CPU oversubscription and ensures that **all application-level parallelism is exclusively managed by the TorcPy runtime scheduler**, allowing the scheduling policy to fully control workload distribution and execution.
