import json
import os

import FWCore.ParameterSet.Config as cms

MODEL_DIR = "HMTntuple/CSCShowerAnalyzer/data"
DEFAULT_TAG = "cm_csc_v1"


def read_norm_stats(stats_path):
    """Classical-mix normalisation from the export sidecar <stem>.json (key "norm", written by
    mds-ml/onnx/export_ragged_cmssw.py from flex_atten/stats/classicalmix_csc.json) into a cms.PSet.
    A bare classicalmix_csc.json is accepted too."""
    stats_path = os.path.expandvars(stats_path)
    with open(stats_path) as f:
        stats = json.load(f)
    stats = stats.get("norm", stats)
    if stats.get("layout") != "csc_classicalmix_v1":
        raise RuntimeError("%s: layout %r, the producer builds csc_classicalmix_v1 inputs"
                           % (stats_path, stats.get("layout")))
    for group in ("simhits_time", "simhits_energy"):
        if stats[group].get("method") != "std":
            raise RuntimeError("%s: '%s' is not a z-score" % (stats_path, group))
    mm, lc, trap = stats["minmax"], stats["layer_centre"], stats["trapezoid"]
    keys = sorted(trap, key=lambda k: tuple(int(x) for x in k.split(",")))   # "station,raw ring"
    if len(lc["min"]) != 3 or len(lc["max"]) != 3:
        raise RuntimeError("%s: layer_centre needs 3 entries" % stats_path)
    return cms.PSet(
        adcMin=cms.double(mm["adc"]["min"]), adcMax=cms.double(mm["adc"]["max"]),
        stripNumberMin=cms.double(mm["strip_number"]["min"]), stripNumberMax=cms.double(mm["strip_number"]["max"]),
        wireGroupMin=cms.double(mm["wire_group"]["min"]), wireGroupMax=cms.double(mm["wire_group"]["max"]),
        wireWidthMin=cms.double(mm["wire_width"]["min"]), wireWidthMax=cms.double(mm["wire_width"]["max"]),
        layerCentreMin=cms.vdouble(*[float(v) for v in lc["min"]]),
        layerCentreMax=cms.vdouble(*[float(v) for v in lc["max"]]),
        trapKey=cms.vint32(*[int(k.split(",")[0]) * 10 + int(k.split(",")[1]) for k in keys]),
        trapVLo=cms.vdouble(*[float(trap[k]["v_lo"]) for k in keys]),
        trapVHi=cms.vdouble(*[float(trap[k]["v_hi"]) for k in keys]),
        trapHwLo=cms.vdouble(*[float(trap[k]["hw_lo"]) for k in keys]),
        trapHwHi=cms.vdouble(*[float(trap[k]["hw_hi"]) for k in keys]),
        timeMean=cms.double(float(stats["simhits_time"]["mean"][0])),
        timeStd=cms.double(float(stats["simhits_time"]["std"][0])),
        energyMean=cms.double(float(stats["simhits_energy"]["mean"][0])),
        energyStd=cms.double(float(stats["simhits_energy"]["std"][0])),
        energyLogEps=cms.double(float(stats["simhits_energy"].get("log_eps", 1e-12))),
    )


mlCSCRecHits = cms.EDProducer(
    "MLCSCRecHitProducer",
    wireDigiLabel=cms.InputTag("muonCSCDigis", "MuonCSCWireDigi"),
    stripDigiLabel=cms.InputTag("muonCSCDigis", "MuonCSCStripDigi"),
    encoderPath=cms.string("%s/mds_hits_%s_encoder.onnx" % (MODEL_DIR, DEFAULT_TAG)),
    decoderPath=cms.string("%s/mds_hits_%s_decoder.onnx" % (MODEL_DIR, DEFAULT_TAG)),
    # kept hits per doc: "mean" = round(count_mu) (offline decode), "upper" = round(mu + slotZ*sd)
    nKeepRule=cms.string("mean"),
    slotZ=cms.double(2.0),   # the run's slot_z
    # fired strip, as the ntupler (cscStripFiredOnly) and the packer: max(ADC) - min(ADC) > 13.3
    stripAdcRangeThreshold=cms.double(13.3),
    minKeep=cms.int32(0),
    minKeepMinMu=cms.double(0.0),
    # filled by read_norm_stats(<stem>.json); the producer throws if left empty
    normalization=cms.PSet(),
    verbose=cms.untracked.bool(False),
)
