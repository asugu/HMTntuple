import json
import os

import FWCore.ParameterSet.Config as cms
# drift time = digi time - the tTrig sync of the standard dt1DRecHits, exactly as the ntupler
from RecoLocalMuon.DTRecHit.dt1DRecHits_LinearDriftFromDB_cfi import dt1DRecHits as _dt1D

MODEL_DIR = "HMTntuple/CSCShowerAnalyzer/data"
DEFAULT_TAG = "dt_cm_v1"


def read_dt_norm(path):
    """DT_NORM scales from the export sidecar <stem>.json (key "norm", mds-ml/onnx/export_dt_cmssw.py)."""
    with open(os.path.expandvars(path)) as f:
        n = json.load(f)["norm"]
    if n.get("max_hits") != 4:
        raise RuntimeError("%s: max_hits %r, the producer packs 4 TDC hits per cell" % (path, n.get("max_hits")))
    return cms.PSet(**{k: cms.double(float(n[k])) for k in
                       ("drift_tscale", "time_scale", "loge_mean", "loge_std", "r_scale", "z_scale")})


mlDTRecHits = cms.EDProducer(
    "MLDTRecHitProducer",
    digiLabel=cms.InputTag("muonDTDigis"),
    tTrigMode=_dt1D.recAlgoConfig.tTrigMode,
    tTrigModeConfig=_dt1D.recAlgoConfig.tTrigModeConfig,
    encoderPath=cms.string("%s/mds_dt_hits_%s_encoder.onnx" % (MODEL_DIR, DEFAULT_TAG)),
    decoderPath=cms.string("%s/mds_dt_hits_%s_decoder.onnx" % (MODEL_DIR, DEFAULT_TAG)),
    nKeepRule=cms.string("mean"),   # round(count_mu), the offline decode (test_dt / event_clustering topk_perdoc)
    slotZ=cms.double(2.0),
    minKeep=cms.int32(0),           # per-doc floor on kept hits, as mlCSCRecHits
    minKeepMinMu=cms.double(0.0),
    normalization=cms.PSet(),       # read_dt_norm(<stem>.json); the producer throws if left empty
    verbose=cms.untracked.bool(False),
)
