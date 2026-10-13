import json
import os

import FWCore.ParameterSet.Config as cms

# ---------------------------------------------------------------------------
# Feature order.
#
# This is the order the network was trained with. It is not alphabetical and
# not source order -- the training pipeline built the column list from a Python
# set union, so the order is whatever that set iteration produced, and it was
# then frozen into every normalized training file and into the weights.
#
# The lists are repeated here (they also live in MLCSCRecHitProducer.cc) so the
# stats can be pulled out of norm_stats_SH_large.json by name rather than by
# position. read_norm_stats() below asserts the two agree, which is the point:
# a future retrain that reorders anything fails loudly at configuration time
# instead of silently feeding the model transposed features.
# ---------------------------------------------------------------------------

# 6 wire/strip (x,y) columns use the layer trapezoid frame; their min/max entries are unused
WIRE_FEATURES = [
    "cscWireDigiTimeBinsOn%d" % i for i in range(16)
] + [
    "cscWireDigiStartX", "cscWireDigiStartY", "cscWireDigiStartZ",
    "cscWireDigiEndX", "cscWireDigiEndY", "cscWireDigiEndZ",
    "cscWireDigiCenterX", "cscWireDigiCenterY", "cscWireDigiCenterZ",
    "cscWireDigiLength", "cscWireDigiWireGroup", "cscWireDigiDetID",
]

STRIP_FEATURES = [
    "cscStripDigiADC_%d" % i for i in range(8)
] + [
    "cscStripStartX", "cscStripStartY", "cscStripStartZ",
    "cscStripEndX", "cscStripEndY", "cscStripEndZ",
    "cscStripDigiLayerX", "cscStripDigiLayerY", "cscStripDigiLayerZ",
    "cscStripLength", "cscStripDigiStripNumber", "cscStripDigiDetID",
]


def read_norm_stats(stats_path):
    """Load minmax stats from norm_stats_SH_*.json into a cms.PSet.

    Reading the file the training wrote, rather than transcribing 96 floats
    into this config, means a retrain is a file swap. The length checks below
    catch the case where a new stats file has a different feature count.
    """
    stats_path = os.path.expandvars(stats_path)
    with open(stats_path) as f:
        stats = json.load(f)

    for group, expected in (("wires", WIRE_FEATURES), ("strips", STRIP_FEATURES)):
        if group not in stats:
            raise RuntimeError("%s: missing '%s' group" % (stats_path, group))
        entry = stats[group]
        method = entry.get("method", "minmax")
        if method != "minmax":
            raise RuntimeError(
                "%s: '%s' uses method '%s'; the producer only implements minmax"
                % (stats_path, group, method))
        for key in ("min", "max"):
            if len(entry[key]) != len(expected):
                raise RuntimeError(
                    "%s: '%s' %s has %d entries, expected %d (feature order changed?)"
                    % (stats_path, group, key, len(entry[key]), len(expected)))

    # output: x,y in cm; t z-scored, E z-scored in log10 (needs pow10)
    for group, method in (("simhits_time", "std"),
                          ("simhits_energy", "std")):
        if group not in stats:
            raise RuntimeError("%s: missing '%s' group" % (stats_path, group))
        got = stats[group].get("method", "minmax")
        if got != method:
            raise RuntimeError(
                "%s: '%s' uses method '%s', the producer inverts '%s'"
                % (stats_path, group, got, method))

    return cms.PSet(
        wireMin=cms.vdouble(*[float(v) for v in stats["wires"]["min"]]),
        wireMax=cms.vdouble(*[float(v) for v in stats["wires"]["max"]]),
        stripMin=cms.vdouble(*[float(v) for v in stats["strips"]["min"]]),
        stripMax=cms.vdouble(*[float(v) for v in stats["strips"]["max"]]),
        timeMean=cms.double(float(stats["simhits_time"]["mean"][0])),
        timeStd=cms.double(float(stats["simhits_time"]["std"][0])),
        energyMean=cms.double(float(stats["simhits_energy"]["mean"][0])),
        energyStd=cms.double(float(stats["simhits_energy"]["std"][0])),
        energyLogEps=cms.double(float(stats["simhits_energy"].get("log_eps", 1e-12))),
    )


MODEL_DIR = "HMTntuple/CSCShowerAnalyzer/data"

mlCSCRecHits = cms.EDProducer(
    "MLCSCRecHitProducer",
    wireDigiLabel=cms.InputTag("muonCSCDigis", "MuonCSCWireDigi"),
    stripDigiLabel=cms.InputTag("muonCSCDigis", "MuonCSCStripDigi"),
    # encoder reports n_slots per doc; decoder runs on exactly those slots
    encoderPath=cms.string("%s/mds_hits_v25abc_encoder.onnx" % MODEL_DIR),
    decoderPath=cms.string("%s/mds_hits_v25abc_decoder.onnx" % MODEL_DIR),
    # kept hits per doc: "mean" = round(count_mu), "upper" = round(mu + slotZ*sd), ~3x more
    nKeepRule=cms.string("mean"),
    # Must equal the run's slot_z; only used by nKeepRule="upper".
    slotZ=cms.double(2.0),
    # Training kept a strip when max(ADC) - min(ADC) exceeded this.
    stripAdcRangeThreshold=cms.double(13.3),
    # Training built docs only from chamber-layers carrying both views.
    requireStripWirePair=cms.bool(True),
    # Minimum fired strips per chamber-layer.
    #
    # MUST equal event_datamaker.py's MIN_FIRED_STRIPS (currently 3), and the
    # ADC threshold above must equal data_maker's STRIP_ADC_THRESHOLD, since
    # "fired" is defined by that cut on both sides. The training documents are
    # built with exactly this rule; a mismatch feeds the model layer
    # occupancies it never saw, with no error to notice.
    #
    # NOTE: this only matches a model trained with --docs strip_wire. A
    # checkpoint trained on the older rechit-gated documents expects a
    # different layer population entirely.
    # Must equal event_datamaker.py's MIN_FIRED_STRIPS (flex_atten/
    # make_event_cluster.py). 2 since the Sep-2026 rebuild.
    minFiredStrips=cms.uint32(2),
    # per-doc floor on kept hits: K = max(minKeep, round(mu)); 0 = plain round(mu)
    minKeep=cms.int32(0),
    minKeepMinMu=cms.double(0.0),   # floor only where count mu >= this
    minKeepME11Only=cms.bool(False),  # floor only ME1/1 docs
    # Filled by read_norm_stats(); the producer throws if left empty.
    normalization=cms.PSet(
        wireMin=cms.vdouble(),
        wireMax=cms.vdouble(),
        stripMin=cms.vdouble(),
        stripMax=cms.vdouble(),
        timeMean=cms.double(0.),
        timeStd=cms.double(1.),
        energyMean=cms.double(0.),
        energyStd=cms.double(1.),
        energyLogEps=cms.double(1e-12),
    ),
    verbose=cms.untracked.bool(False),
)
