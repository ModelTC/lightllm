from .continues_batch.impl import ContinuesBatchQueue
from .continues_batch.beam_impl import BeamContinuesBatchQueue
from .continues_batch.pd_decode_impl import QueueForPDDecode
from .chunked_prefill.impl_for_pd_prefill import QueueForPDChunkedPrefill
from .chunked_prefill.impl import ChunkedPrefillQueue
from .dp_base_queue import DpQueue


def _get_req_queue_class(args, router, dp_size_in_node: int):
    if args.diverse_mode:
        return BeamContinuesBatchQueue
    if args.token_healing_mode:
        return ContinuesBatchQueue
    if args.output_constraint_mode != "none":
        return ContinuesBatchQueue
    if args.first_token_constraint_mode:
        return ContinuesBatchQueue
    if args.run_mode == "decode":
        return QueueForPDDecode
    if args.run_mode == "prefill":
        return QueueForPDChunkedPrefill

    if args.disable_chunked_prefill:
        return ContinuesBatchQueue
    else:
        return ChunkedPrefillQueue


def build_req_queue(args, router, dp_size_in_node: int):
    queue_class = _get_req_queue_class(args, router, dp_size_in_node)

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
    else:
        return DpQueue(args, router, queue_class, dp_size_in_node)
