import os, sys

sys.path.append(os.getcwd())

from SCHOOL.staffroom.calligraphist import S_OUTPUT

class DummyCtx:
    def __enter__(self):
        return lambda *args, **kwargs: None
    def __exit__(self, exc_type, exc, tb):
        pass

class DummyCounsellor:
    def infodump(self, label):
        return DummyCtx()

def test_group_stats_order():
    calli = S_OUTPUT(DummyCounsellor())
    stats = {
        '1E_0_vector_norm': 1.0,
        '5A_0_attnOut_norm': 2.0,
        '2N_1_normedInput_norm': 3.0,
        '3INN_cerebellumMean': 4.0,
        '4A_memory_4M_0_rawActs_norm': 5.0,
        '4B_memory2_4M_0_rawActs_norm': 6.0,
        '6L_logitMax': 7.0,
    }
    grouped = calli.groupStatsBySection(stats)
    labels = [label for label, _ in grouped]
    assert labels == [
        'EMBED STATS',
        'ATTENTION STATS',
        'NEURON STATS',
        'INTERNEURON STATS',
        'MEMORY STATS',
        'MEMORY2 STATS',
        'LOGIT STATS',
    ]
