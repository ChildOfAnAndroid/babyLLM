"""Construct an isolated checkpoint model for explicitly requested diagnostics."""


def build_model(window_max):
    from babyLLM import BABYLLM
    from config import modelDevice, learningRateGOAL
    from school.staffroom.counsellor import COUNSELLOR
    from school.staffroom.librarian import LIBRARIAN
    from school.staffroom.calligraphist import S_OUTPUT
    from school.staffroom.HE_IS_SCRIBE import SCRIBE

    counsellor = COUNSELLOR("probe", _debug=False, _durations=False)
    librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
    calligraphist = S_OUTPUT(_counsellor=counsellor)
    scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                    _librarian=librarian, _numTokensPerStep=window_max)
    model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                   _scribe=scribe, _librarian=librarian, _device=modelDevice,
                   _numTokensPerStep=window_max, _first=True,
                   _learningRateGOAL=learningRateGOAL)
    model.loadModel()
    model.to(modelDevice)
    model.eval()
    return model, librarian
