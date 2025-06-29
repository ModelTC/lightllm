import torch
from typing import Callable
from typing import List


def error(y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)


    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    y_pred = torch.flatten(y_pred).float()
    y_real = torch.flatten(y_real).float()

    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute snr loss for tensors with different shape. " f"({y_pred.shape} and {y_real.shape})"
        )

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)
    return snr.item()


def benchmark(func: Callable, shape: List[int], tflops: float, steps: int, *args, **kwargs):
    """
    A decorator function to assist in performance testing of CUDA operations.

    This function will:
    1. Automatically determine whether any parameters in the argument list,
       or the output of the `func`, are of type `torch.Tensor`.
    2. If so, calculate the memory usage of the input and output tensors
       on the GPU (based on their data type and `torch.numel()`).
    3. Establish a CUDA graph and attempt to execute `func` repeatedly for `steps` iterations.
    4. Record the execution time during these iterations.
    5. Use the information above to compute the compute performance (TFLOPS) and memory throughput.

    Args:
        func (function): The function to benchmark.
        shape (list of int): The problem shape.
        tflops (float): The computational workload (in TFLOPS) per call of `func`.
        steps (int): The number of times the function is executed during benchmarking.
        *args: Positional arguments to be passed to the `func`.
        **kwargs: Keyword arguments to be passed to the `func`.

    Returns:
        function result
    """

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    # Check for torch.Tensor in inputs and outputs
    input_tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    input_tensors += [value for value in kwargs.values() if isinstance(value, torch.Tensor)]

    def calculate_memory(tensor: torch.Tensor):
        """Calculate memory usage in bytes for a tensor."""
        return tensor.numel() * tensor.element_size()

    input_memory = sum(calculate_memory(t) for t in input_tensors)

    # Execute the function to inspect outputs
    with torch.no_grad():
        output = func(*args, **kwargs)

    output_memory = 0
    if isinstance(output, torch.Tensor):
        output_memory = calculate_memory(output)
    elif isinstance(output, (list, tuple)):
        output_memory = sum(calculate_memory(o) for o in output if isinstance(o, torch.Tensor))

    total_memory = input_memory + output_memory

    # Warm-up and CUDA graph creation
    for _ in range(10):  # Warm-up
        func(*args, **kwargs)

    torch.cuda.synchronize()  # Ensure no pending operations

    # Benchmark the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()  # Ensure all operations are finished
    elapsed_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds

    # Calculate performance metrics
    elapsed_time_s = elapsed_time_ms / 1000  # Convert to seconds
    avg_time_per_step = elapsed_time_s / steps
    compute_performance = tflops / avg_time_per_step  # TFLOPS
    memory_throughput = (total_memory * steps / (1024 ** 3)) / elapsed_time_s  # GB/s

    # Print performance metrics
    print(f"Function: {func.__name__}{shape}")
    # print(f"Function: {func.__ne__}{shape}")
    print(f"Elapsed Time (total): {elapsed_time_s:.4f} seconds")
    print(f"Average Time Per Step: {avg_time_per_step * 1000 :.3f} ms")
    print(f"Compute Performance: {compute_performance:.2f} TFLOPS")
    print(f"Memory Throughput: {memory_throughput:.2f} GB/s")
    print("")  # print a blank line.
