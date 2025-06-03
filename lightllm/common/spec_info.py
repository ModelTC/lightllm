from enum import IntEnum, auto


class SpeculativeDecodeAlgorithm(IntEnum):
    NONE = auto()
    MTP = auto()
    MTP_MOUDLE = auto()

    def is_none(self):
        return self == SpeculativeDecodeAlgorithm.NONE

    def is_mtp(self):
        return self == SpeculativeDecodeAlgorithm.MTP

    def is_mtp_module(self):
        return self == SpeculativeDecodeAlgorithm.MTP_MOUDLE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "MTP": SpeculativeDecodeAlgorithm.MTP,
            "MTP_MOUDLE": SpeculativeDecodeAlgorithm.MTP_MOUDLE,
            "NONE": SpeculativeDecodeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]

    def decode_len(self):
        if self == SpeculativeDecodeAlgorithm.NONE:
            return 1
        if self == SpeculativeDecodeAlgorithm.MTP:
            return 2
        if self == SpeculativeDecodeAlgorithm.MTP_MOUDLE:
            return 2
