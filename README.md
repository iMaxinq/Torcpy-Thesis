# Distributed Video Denoising using TorcPy and MPI

This repository contains the software implementation developed as part of my thesis. It presents a distributed video processing framework, specifically targeting video denoising, utilizing the **Message Passing Interface (MPI)** via the **mpi4py** binding, alongside a custom task-scheduling runtime system designated as **TorcPy**.

The system is designed to ingest a video file, extract its constituent frames, distribute the workload across multiple nodes of a computational cluster for parallel processing (leveraging OpenCV for the denoising algorithms), and ultimately aggregate the processed frames into a synthesized output.

---

## Prerequisites

To deploy and execute this framework within a distributed cluster environment, the following dependencies are required:

* A **Linux-based Operating System** (e.g., Ubuntu/Debian distributions) on all constituent nodes.
* **Python 3.7** or higher.
* A functional installation of **MPICH** or **OpenMPI**.

---

## System Setup and Network Configuration

To facilitate seamless inter-node communication and process spawning by the MPI daemon without manual intervention, establishing **passwordless SSH authentication** across all machines within the cluster is strictly mandated.

### 1. Passwordless SSH Configuration

On the primary execution node (Master Node), generate an SSH key pair:

```bash
ssh-keygen -t rsa -b 4096
(Depress the Enter key at all prompts to enforce an empty passphrase).Subsequently, propagate the public key to all participating nodes in the network architecture (including the master node itself, if it is designated to operate as a worker):Bashssh-copy-id user@<NODE_1_IP>
ssh-copy-id user@<NODE_2_IP>
# Replicate this procedure for all constituent nodes...
Verify the configuration by initiating an SSH session (ssh user@<NODE_IP>). The absence of a password prompt indicates successful cryptographic authentication setup.2. System-Level DependenciesExecute the following package management commands on all nodes to install MPICH and the requisite Python development tools:Bashsudo apt-get update
sudo apt-get install -y mpich libmpich-dev python3 python3-pip python3-venv libgl1-mesa-glx
(The libgl1-mesa-glx library is a mandatory dependency for ensuring the structural integrity of OpenCV operations in headless server environments).3. Python Environment InitializationIt is highly recommended to isolate the project dependencies utilizing a virtual environment. Navigate to the repository directory and execute:Bashpython3 -m venv venv
source venv/bin/activate
Proceed to install the necessary Python libraries:Bashpip install mpi4py opencv-python termcolor coloredlogs
Experimental ExecutionPrior to execution, ensure that the input artifact input_noisy.mp4 resides in the root execution directory alongside the source code.Hostfile GenerationConstruct a configuration file named hostfile (or machinefile). This file must detail the IP addresses of the cluster nodes and the corresponding number of MPI slots (processes) allocated to each.Example hostfile configuration:Plaintextlocalhost slots=1
192.168.1.101 slots=1
192.168.1.102 slots=1
Execution CommandTo initiate the distributed computation, invoke the mpiexec (or mpirun) command, supplying the hostfile and specifying the aggregate number of nodes (-n):Bashmpiexec -n 3 -f hostfile python3 main.py
(Modify main.py to match the exact nomenclature of your primary Python script).Upon successful termination, the algorithmically denoised frames will be systematically archived in the ./output_frames directory.Runtime Configuration and Task Scheduling (TorcPy)The TorcPy runtime environment parameters are modulated dynamically via Environment Variables. These must be exported prior to the invocation of the mpiexec command.The configurable parameters are detailed below:VariableDescriptionDefaultPermitted ValuesTORCPY_SCHEDULINGDictates the algorithmic policy for task distribution across nodes.round_robinround_robin, weighted, heftTORCPY_WORKERSDefines the concurrency level (worker threads) per MPI process.1Integer (e.g., 4, 8)TORCPY_STEALINGToggles the distributed Work-Stealing protocol for dynamic load balancing.FalseTrue, FalseTORCPY_PRINT_STATSEnables the terminal output of detailed diagnostic metrics (scheduling overheads, data transfer volumes, etc.) post-execution.FalseTrue, FalseExecution paradigm utilizing HEFT scheduling, configuring 4 concurrent workers per node, and enabling diagnostic output:Bashexport TORCPY_SCHEDULING="heft"
export TORCPY_WORKERS=4
export TORCPY_PRINT_STATS="True"

mpiexec -n 3 -f hostfile python3 main.py
Critical Note on CPU Utilization and ThreadingThe codebase explicitly restricts the underlying C-based libraries (OpenBLAS, MKL, OpenCV) to single-threaded execution through os.environ overrides. This architectural constraint is purposefully engineered to mitigate the deleterious effects of CPU oversubscription, thereby delegating the exclusive management of computational parallelization to the TorcPy scheduling heuristic.
