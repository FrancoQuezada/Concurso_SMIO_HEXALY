"""Repair operator interfaces for the future ALNS phase."""


class RepairOperator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Repair operators are planned for a later phase")
