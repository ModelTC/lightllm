import time
from prometheus_client import Histogram, Counter, Gauge
monitor_registry = {}
monitor_info = {
    "lightllm_request_count": "The total number of requests",
    "lightllm_request_success": "The total number of requests",
    "lightllm_request_duration": "Duration of the request",
    "lightllm_request_validation_duration": "Validation time of the request",
    "lightllm_request_queue_duration": "Queue time of the request",
    "lightllm_request_inference_duration": "Inference time of the request",
    "lightllm_request_mean_time_per_token_duration": "Per token time of the request",
    "lightllm_request_first_token_duration": "First token time of the request",
    "lightllm_request_input_length": "Length of the input tokens",
    "lightllm_request_generated_tokens": "Number of generated tokens",
    "lightllm_request_max_new_tokens": "Max new token",
    "lightllm_batch_next_size": "Batch size",
    "lightllm_batch_current_size": "Current batch size",
    "lightllm_batch_pause_size": "The number of pause requests",
    "lightllm_queue_size": "Queue size"
}

def histogram_timer(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time() 
            result = func(*args, **kwargs)  
            elapsed_time = time.time() - start_time  
            monitor_registry[name].observe(elapsed_time) 
            return result
        return wrapper
    return decorator

def init_api_server_monitor(args):
    duration_buckets = []
    value = 0.001
    n_duration_buckets = 35
    for _ in range(n_duration_buckets):
        value *= 1.5
        duration_buckets.append(value)
    
    for k, v in monitor_info.items():
        if "duration" in k:
            histogram = Histogram(k, v, buckets=duration_buckets)
            monitor_registry[k] = histogram

    max_req_total_len = args.max_req_total_len
    generate_tokens_buckets = [max_req_total_len / 100. * (i + 1) for i in range(0, 100)]
    create_histogram("lightllm_request_generated_tokens", generate_tokens_buckets)

    create_counter("lightllm_request_count")
    create_counter("lightllm_request_success")

def init_httpserver_monitor(args):
    max_req_input_len = args.max_req_input_len
    input_len_buckets = [max_req_input_len / 100. * (i + 1) for i in range(0, 100)]
    create_histogram("lightllm_request_input_length", input_len_buckets)
    
    max_req_total_len = args.max_req_total_len
    generate_tokens_buckets = [max_req_total_len / 100. * (i + 1) for i in range(0, 100)]
    create_histogram("lightllm_request_max_new_tokens", generate_tokens_buckets)

def init_router_monitor():
    create_gauge("lightllm_queue_size")
    create_gauge("lightllm_batch_current_size")
    create_gauge("lightllm_batch_pause_size")
    batch_size_buckets = [i + 1 for i in range(0, 1024)]
    create_histogram("lightllm_batch_next_size", batch_size_buckets)

def create_histogram(name, buckets):
    histogram = Histogram(name, monitor_info[name], buckets=buckets)
    monitor_registry[name] = histogram 

def create_counter(name):
    histogram = Counter(name, monitor_info[name])
    monitor_registry[name] = histogram 

def create_gauge(name):
    gauge = Gauge(name,  monitor_info[name])
    monitor_registry[name] = gauge 

def counter_inc(name):
    monitor_registry[name].inc()

def histogram_observe(name, value):
    monitor_registry[name].observe(value) 

def gauge_set(name, value):
    monitor_registry[name].set(value) 