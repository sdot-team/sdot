from dataclasses import dataclass

@dataclass
class IoCategory:
    want_return : bool
    want_output : bool
    want_input : bool
