import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# CSC clusters
from RecoMuon.MuonRechitClusterProducer.cscRechitClusterProducer_cfi import cscRechitClusterProducer
from RecoMuon.MuonRechitClusterProducer.dtRechitClusterProducer_cfi import dtRechitClusterProducer 

cscMDSClusterTable = cms.EDProducer("SimpleMuonRecHitClusterFlatTableProducer",
    src  = cms.InputTag("ca4CSCrechitClusters"),
    name = cms.string("cscMDSHLTCluster"),
    doc  = cms.string("MDS cluster at HLT"),
    variables = cms.PSet(
        eta = Var("eta", float, doc="cluster eta"),
        phi = Var("phi", float, doc="cluster phi"),
        x   = Var("x"  , float, doc="cluster x"),
        y   = Var("y"  , float, doc="cluster y"),
        z   = Var("z"  , float, doc="cluster z"),
        r   = Var("r"  , float, doc="cluster r"),
        size   = Var("size"  , int, doc="cluster size"),
        nStation   = Var("nStation"  , int, doc="cluster nStation"),
        avgStation   = Var("avgStation"  , float, doc="cluster avgStation"),
        nMB1   = Var("nMB1"  , int, doc="cluster nMB1"),
        nMB2   = Var("nMB2"  , int, doc="cluster nMB2"),
        nME11   = Var("nME11"  , int, doc="cluster nME11"),
        nME12   = Var("nME12"  , int, doc="cluster nME12"),
        nME41   = Var("nME41"  , int, doc="cluster nME41"),
        nME42   = Var("nME42"  , int, doc="cluster nME42"),
        time   = Var("time"  , float, doc="cluster time = avg cathode and anode time"),
        timeSpread   = Var("timeSpread"  , float, doc="cluster timeSpread")
    )
)

dtMDSClusterTable = cscMDSClusterTable.clone(
    src = cms.InputTag("ca4DTrechitClusters"),
    name= cms.string("dtMDSHLTCluster")
)

from HMTntuple.CSCShowerAnalyzer.cscShowerDigiTable_cfi import cscShowerDigiTable
from HMTntuple.CSCShowerAnalyzer.cscMDSshowerTable_cfi import cscMDSshowerTable 
from HMTntuple.CSCShowerAnalyzer.dtMDSshowerTable_cfi import dtMDSshowerTable 

dataCSCdigiTable = cscShowerDigiTable.clone(
    LCTShower = cms.InputTag("muonCSCDigis","MuonCSCShowerDigi"),
    name = cms.string("lct")
)
emulCSCdigiTable = cscShowerDigiTable.clone(
    LCTShower =  cms.InputTag("cscTriggerPrimitiveDigis"),
    name = cms.string("elct")
)

from HMTntuple.CSCShowerAnalyzer.cscRechitsTable_cfi import cscRechitsTable 
from HMTntuple.CSCShowerAnalyzer.dtRechitsTable_cfi import dtRechitsTable 
from HMTntuple.CSCShowerAnalyzer.rpcRechitsTable_cfi import rpcRechitsTable 
from HMTntuple.CSCShowerAnalyzer.cscSegmentsTable_cfi import cscSegmentsTable 
from HMTntuple.CSCShowerAnalyzer.dtSegmentsTable_cfi import dtSegmentsTable 

cscMDSshowerTable = cscMDSshowerTable.clone( 
    name = cms.string("cscMDSCluster"),
    recHitLabel = cms.InputTag("csc2DRecHits"),
    segmentLabel = cms.InputTag("dt4DSegments"),
    rpcLabel = cms.InputTag("rpcRecHits")
)
dtMDSshowerTable = dtMDSshowerTable.clone( 
    name = cms.string("dtMDSCluster"),
    recHitLabel = cms.InputTag("dt1DRecHits"),
    rpcLabel = cms.InputTag("rpcRecHits")
)

cscRechitsTable = cscRechitsTable.clone( 
    recHitLabel = cms.InputTag("csc2DRecHits")
)
dtRechitsTable = dtRechitsTable.clone( 
    recHitLabel = cms.InputTag("dt1DRecHits")
)
rpcRechitsTable = rpcRechitsTable.clone( 
    recHitLabel = cms.InputTag("rpcRecHits")
)
cscSegmentsTable = cscSegmentsTable.clone( 
    segmentLabel = cms.InputTag("cscSegments")
)
dtSegmentsTable = dtSegmentsTable.clone( 
    segmentLabel = cms.InputTag("dt4DSegments")
)


def add_mdsTables(process, MDSshowerDigi=False, saveRechits=False,
                  MLrechits=False, saveMLrechits=False, MLstatsPath=None,
                  looseNRechitMin=0, MLmodelTag=None,
                  MLdtRechits=False, MLdtModelTag=None, MLjointClusters=False):
    # MLdtRechits: DT twin of MLrechits (mlDTRecHits -> DTRecHitCollection, needs muonDTDigis).
    # MLjointClusters: one CA clustering over ML CSC + ML DT hits together (muonMDSClusterML table);
    # needs both. MLstatsPath / MLmodelTag: CSC graphs + sidecar data/mds_hits_<tag>{_encoder,_decoder}.onnx/.json
    # looseNRechitMin > 0 adds a SECOND pair of offline shower tables with the
    # cluster hit threshold lowered from the default 50. It measures the
    # efficiency CEILING: the nominal denominator is purely geometric (LLP
    # decayed inside the CSC volume), so it charges the reconstruction for
    # decays that never left enough energy to cluster at all. A loose run
    # separates that physics loss from the algorithmic one --
    #   ceiling  = fiducial LLPs with a LOOSE cluster within dR<0.4
    #   nominal  = fiducial LLPs with a 50-hit cluster within dR<0.4
    # and the gap between them is what no clusterer could have recovered.
    #
    # nRechitMin is a cut applied AFTER the FastJet CA clustering
    # (cscMDSshowerTableProducer.cc:188), so the loose collection is a strict
    # SUPERSET of the nominal one -- same clusters, plus the small ones. That
    # is what makes the two directly comparable in the same file.

    # produce the tables for offline clusters 
    process.cscMDSshowerTable = cscMDSshowerTable    
    process.dtMDSshowerTable = dtMDSshowerTable   
    # produce the HLT clusters 
    process.ca4CSCrechitClusters = cscRechitClusterProducer    
    process.ca4DTrechitClusters = dtRechitClusterProducer
    # produce the tables for HLT clusters 
    process.cscMDSClusterTable = cscMDSClusterTable
    process.dtMDSClusterTable = dtMDSClusterTable 

    process.MDSTask = cms.Task(process.ca4CSCrechitClusters)
    process.MDSTask.add(process.ca4DTrechitClusters)
    process.MDSTask.add(process.cscMDSClusterTable)
    process.MDSTask.add(process.dtMDSClusterTable)
    process.MDSTask.add(process.cscMDSshowerTable)
    process.MDSTask.add(process.dtMDSshowerTable)

    if looseNRechitMin > 0:
        process.cscMDSshowerTableLoose = cscMDSshowerTable.clone(
            name = cms.string("cscMDSClusterLoose"),
            nRechitMin = cms.int32(looseNRechitMin),
        )
        process.MDSTask.add(process.cscMDSshowerTableLoose)
        process.dtMDSshowerTableLoose = dtMDSshowerTable.clone(
            name = cms.string("dtMDSClusterLoose"),
            nRechitMin = cms.int32(looseNRechitMin),
        )
        process.MDSTask.add(process.dtMDSshowerTableLoose)

    if MDSshowerDigi:
        process.load("CalibMuon.CSCCalibration.CSCL1TPLookupTableEP_cff")
        process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
        process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
        process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
        process.cscTriggerPrimitiveDigis.commonParam.runME11ILT = False
        process.cscTriggerPrimitiveDigis.commonParam.runME21ILT = False
        process.cscTriggerPrimitiveDigis.showerParam.cathodeShower.showerThresholds = cms.vuint32(
            # ME1/1
            100, 100, 100,
            # ME1/2
            10000, 10000, 10000,
            # ME1/3
            10000, 10000, 10000,
            # ME2/1
            6, 33, 35,
            # ME2/2
            10000, 10000, 10000,
            # ME3/1
            6, 31, 33,
            # ME3/2
            10000, 10000, 10000,
            # ME4/1
            6, 34, 36,
            # ME4/2
            10000, 10000, 10000
        ) 
        process.cscTriggerPrimitiveDigis.showerParam.anodeShower.showerThresholds = cms.vuint32(
            # ME1/1
            140, 140, 140,
            # ME1/2
            140, 140, 140,
            # ME1/3
            7, 14, 18,
            # ME2/1
            12, 56, 58,
            # ME2/2
            12, 28, 32,
            # ME3/1
            12, 55, 57,
            # ME3/2
            12, 26, 34,
            # ME4/1
            12, 62, 64,
            # ME4/2
            12, 27, 31
        ) 
        process.dataCSCdigiTable = dataCSCdigiTable 
        process.emulCSCdigiTable = emulCSCdigiTable 

        process.MDSTask.add(process.cscTriggerPrimitiveDigis)
        process.MDSTask.add(process.dataCSCdigiTable)
        process.MDSTask.add(process.emulCSCdigiTable)

    if saveRechits:
        process.cscRechitsTable = cscRechitsTable
        process.dtRechitsTable = dtRechitsTable
        process.rpcRechitsTable = rpcRechitsTable
        process.cscSegmentsTable = cscSegmentsTable 
        process.dtSegmentsTable = dtSegmentsTable 
        process.MDSTask.add(process.cscRechitsTable)
        process.MDSTask.add(process.dtRechitsTable)
        process.MDSTask.add(process.rpcRechitsTable)
        process.MDSTask.add(process.cscSegmentsTable)
        process.MDSTask.add(process.dtSegmentsTable)

    if MLrechits:
        # ML rechits: run the MDS hit-regression network on CSC wire/strip
        # digis and emit a CSCRecHit2DCollection. Because the output is a
        # normal CSCRecHit2DCollection, the *existing* cluster producer and
        # cluster table work on it unchanged -- a clone with a different
        # recHitLabel is the whole integration.
        #
        # Needs digis, so this only works on RAW (see MDSshowerDigi above for
        # the unpacker). On AOD there are no wire/strip digis to read.
        import os
        from HMTntuple.CSCShowerAnalyzer.mlCSCRecHits_cfi import (
            mlCSCRecHits, read_norm_stats, MODEL_DIR, DEFAULT_TAG)

        process.mlCSCRecHits = mlCSCRecHits
        # normalisation travels with the graphs: the export sidecar data/mds_hits_<tag>.json
        process.mlCSCRecHits.normalization = read_norm_stats(MLstatsPath or os.path.join(
            os.environ["CMSSW_BASE"], "src", MODEL_DIR, "mds_hits_%s.json" % (MLmodelTag or DEFAULT_TAG)))
        # MLmodelTag selects data/mds_hits_<tag>_{encoder,decoder}.onnx
        if MLmodelTag:
            process.mlCSCRecHits.encoderPath = cms.string(
                "%s/mds_hits_%s_encoder.onnx" % (MODEL_DIR, MLmodelTag))
            process.mlCSCRecHits.decoderPath = cms.string(
                "%s/mds_hits_%s_decoder.onnx" % (MODEL_DIR, MLmodelTag))

        process.ca4CSCrechitClustersML = cscRechitClusterProducer.clone(
            recHitLabel = "mlCSCRecHits"
        )
        process.cscMDSClusterTableML = cscMDSClusterTable.clone(
            src  = cms.InputTag("ca4CSCrechitClustersML"),
            name = cms.string("cscMDSClusterML"),
            doc  = cms.string("MDS cluster built from ML-regressed CSC rechits. "
                              "Time = ML hit time, TOF - OFF ns (prompt muon = 0), not the "
                              "rechit convention: do not port time cuts from standard clusters.")
        )

        # --- offline-style ML clusters -------------------------------------
        # The block above is the *HLT* clusterer (ca4 / cscRechitClusterProducer).
        # cscMDSshowerTable is the OFFLINE path, and it is likewise driven by an
        # InputTag -- cscMDSshowerTableProducer takes `recHitLabel`, does its own
        # clustering internally, and emits TWO tables: `<name>` (cluster) and
        # `<name>Rechits` (the constituent hits). So the offline twin is a clone
        # as well, same argument as for the HLT one: no parallel infrastructure.
        #
        # This is the table that lets you look INSIDE an ML cluster. The HLT
        # table carries cluster kinematics only; without this, ML clusters could
        # not be compared to standard ones hit-by-hit.
        #
        # Reads only localPosition / cscDetId / tpeak / wireTime for the
        # clustering and every cluster-level variable -- all four are genuinely
        # filled by MLCSCRecHitProducer -- plus energyDepositedInLayer, which is
        # the model's E. Those are all physics.
        #
        # NOT physics on ML hits, because MLCSCRecHitProducer has no strip or
        # wire-group identity to report and passes placeholders (see the
        # `channels{0}` / empty `adcs` / `wgroups{0}` comment there). MEASURED
        # on 8724 ML cluster hits, DY FEVTSIM, 2026-08-19 -- each is 100%
        # constant:
        #     <name>Rechits_NStrips      always 1
        #     <name>Rechits_NWireGroups  always 1
        #     <name>Rechits_HitWire      always 0
        #     <name>Rechits_WGroupsBX    always 0
        #     <name>Rechits_Quality      always 0
        #
        # These are 0/1, NOT -999 sentinels, so they do not look wrong at a
        # glance. To tell an ML hit from a real one, test `HitWire == 0`: CSC
        # wire numbering is 1-based, and 0 of 37273 real cluster hits have it.
        # Do NOT use NStrips==1 (0.85% of real hits) or Quality==0 (0.44%) --
        # both occur legitimately.
        #
        # E runs the OTHER WAY, which is easy to get backwards. On ML hits it is
        # the model's regressed energy and is real (1.7e-6 .. 2.8e-4, median
        # 2.7e-6). On STANDARD rechits it is the dummy -998 for every hit --
        # `energyDepositedInLayer()` is not filled by csc2DRecHits, hence the
        # "//not saved" comment in cscMDSshowerTableProducer.cc. So E is
        # meaningful only on the ML side, and cannot be compared between them.
        #
        # Time caveat compounds here: the offline producer prunes on the spread
        # between tpeak and wireTime (pruneCut, stripErr/wireError), but both are
        # filled from the model's single `t`. They differ only by the 0.01 ns
        # quantization of wireTime's integer storage (measured max |diff|
        # 5.0e-3), so that two-sided weighting has no real anode/cathode
        # disagreement to act on. Visible downstream: cluster timeSpread comes
        # out at median 1.13 ns for ML vs 14.49 ns for standard clusters on the
        # same events. Any selection on timeSpread will behave completely
        # differently on the two -- do not port a cut across.
        process.cscMDSshowerTableML = cscMDSshowerTable.clone(
            recHitLabel = cms.InputTag("mlCSCRecHits"),
            name = cms.string("cscMDSClusterMLOffline"),
        )

        # The ML loose twin. Both loose tables must use the SAME threshold or
        # the two ceilings are not comparable, which is the whole point.
        if looseNRechitMin > 0:
            process.cscMDSshowerTableMLLoose = cscMDSshowerTable.clone(
                recHitLabel = cms.InputTag("mlCSCRecHits"),
                name = cms.string("cscMDSClusterMLOfflineLoose"),
                nRechitMin = cms.int32(looseNRechitMin),
            )
            process.MDSTask.add(process.cscMDSshowerTableMLLoose)

        process.MDSTask.add(process.mlCSCRecHits)
        process.MDSTask.add(process.ca4CSCrechitClustersML)
        process.MDSTask.add(process.cscMDSClusterTableML)
        process.MDSTask.add(process.cscMDSshowerTableML)

        if saveMLrechits:
            # Kept opt-in, mirroring saveRechits: CSC rechits are ~5 kb/event
            # and ML hits are a second such collection.
            process.mlCSCRechitsTable = cscRechitsTable.clone(
                recHitLabel = cms.InputTag("mlCSCRecHits"),
                name = cms.string("cscMLRechits")
            )
            process.MDSTask.add(process.mlCSCRechitsTable)

    if MLdtRechits:
        # DT twin: ML DT hits as DTRecHit1DPair (left = right = predicted local x, y), so the stock DT
        # clusterer and the DT tables run on a clone. ca4 (HLT) uses the CHAMBER frame for DT local
        # positions, like it does for real rechits; dtMDSshowerTable (offline) uses the layer frame.
        import os
        from HMTntuple.CSCShowerAnalyzer.mlDTRecHits_cfi import mlDTRecHits, read_dt_norm, MODEL_DIR as DT_DIR, \
            DEFAULT_TAG as DT_TAG
        tag = MLdtModelTag or DT_TAG
        process.mlDTRecHits = mlDTRecHits.clone(
            encoderPath = cms.string("%s/mds_dt_hits_%s_encoder.onnx" % (DT_DIR, tag)),
            decoderPath = cms.string("%s/mds_dt_hits_%s_decoder.onnx" % (DT_DIR, tag)),
            normalization = read_dt_norm(os.path.join(os.environ["CMSSW_BASE"], "src", DT_DIR,
                                                      "mds_dt_hits_%s.json" % tag)),
        )
        process.ca4DTrechitClustersML = dtRechitClusterProducer.clone(recHitLabel = "mlDTRecHits")
        process.dtMDSClusterTableML = dtMDSClusterTable.clone(
            src  = cms.InputTag("ca4DTrechitClustersML"),
            name = cms.string("dtMDSClusterML"),
            doc  = cms.string("MDS cluster built from ML-regressed DT hits (stock DT clusterer)"),
        )
        # hitInfoLabel: per-hit energy / unquantised time, which DTRecHit1DPair cannot carry
        process.dtMDSshowerTableML = dtMDSshowerTable.clone(
            recHitLabel = cms.InputTag("mlDTRecHits"),
            hitInfoLabel = cms.string("mlDTRecHits"),
            name = cms.string("dtMDSClusterMLOffline"),
        )
        for m in ("mlDTRecHits", "ca4DTrechitClustersML", "dtMDSClusterTableML", "dtMDSshowerTableML"):
            process.MDSTask.add(getattr(process, m))
        if looseNRechitMin > 0:
            process.dtMDSshowerTableMLLoose = process.dtMDSshowerTableML.clone(
                name = cms.string("dtMDSClusterMLOfflineLoose"),
                nRechitMin = cms.int32(looseNRechitMin),
            )
            process.MDSTask.add(process.dtMDSshowerTableMLLoose)
        if saveMLrechits:
            process.mlDTRechitsTable = dtRechitsTable.clone(
                recHitLabel = cms.InputTag("mlDTRecHits"),
                hitInfoLabel = cms.string("mlDTRecHits"),
                name = cms.string("dtMLRechits")
            )
            process.MDSTask.add(process.mlDTRechitsTable)

    if MLjointClusters:
        if not (MLrechits and MLdtRechits):
            raise RuntimeError("MLjointClusters needs MLrechits=True and MLdtRechits=True")
        from HMTntuple.CSCShowerAnalyzer.muonMDSClusterTable_cfi import muonMDSClusterTable
        process.muonMDSClusterTableML = muonMDSClusterTable.clone(
            cscRecHitLabel = cms.InputTag("mlCSCRecHits"),
            dtRecHitLabel = cms.InputTag("mlDTRecHits"),
            name = cms.string("muonMDSClusterML"),
            saveHits = cms.bool(saveMLrechits),
        )
        process.MDSTask.add(process.muonMDSClusterTableML)
        if looseNRechitMin > 0:
            process.muonMDSClusterTableMLLoose = process.muonMDSClusterTableML.clone(
                name = cms.string("muonMDSClusterMLLoose"),
                nRechitMin = cms.int32(looseNRechitMin),
                saveHits = cms.bool(False),
            )
            process.MDSTask.add(process.muonMDSClusterTableMLLoose)

    process.nanoTableTaskCommon.add(process.MDSTask)

    return process
