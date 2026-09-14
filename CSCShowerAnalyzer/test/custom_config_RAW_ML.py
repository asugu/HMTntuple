# MDSnano on RAW with ML rechits.
#
# RAW -> unpack CSC digis -> MLCSCRecHitProducer -> cluster -> nano tables.
#
# Follows runCSCShowerAnalyzer_cfg.py for the RAW/unpacker part and
# custom_config_AOD_data.py for the nano part; the ML side is the MLrechits
# flag in custom_mds_cff.add_mdsTables, not a parallel path.
#
# Run with e.g.
#   cmsRun custom_config_RAW_ML.py \
#       inputFiles=/store/data/Run2024I/ZeroBias/RAW/... maxEvents=10

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register('statsPath', '',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'Path to norm_stats_SH_large.json used to train the ONNX graphs')
options.register('saveMLrechits', False,
                 VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 'Also store the ML rechits themselves, not just the clusters')
options.register('globalTag', 'auto:run3_data',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'Global tag')
options.parseArguments()

process = cms.Process('MDSNANO', Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# MLCSCRecHitProducer reports its per-bucket event tally from endStream() as
# LogInfo, which the default threshold drops. limit=-1 lifts the per-category
# occurrence cap. Verbosity only.
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.MLCSCRecHitProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.source = cms.Source(
    'PoolSource',
    fileNames=cms.untracked.vstring(options.inputFiles),
    inputCommands=cms.untracked.vstring('keep *'),
)
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.maxEvents),
    output=cms.optional.untracked.allowed(cms.int32, cms.PSet),
)

# --- RAW -> digis -----------------------------------------------------------
# Only the CSC unpacker is needed: the ML producer eats wire and strip digis.
process.load('EventFilter.CSCRawToDigi.cscUnpacker_cfi')

# The CSC unpacker needs these two ES products, which the plain RAW workflow
# does not pull in on its own (same boilerplate as runCSCShowerAnalyzer_cfg.py).
process.cscIndexerSource = cms.ESSource(
    'EmptyESSource',
    recordName=cms.string('CSCIndexerRecord'),
    firstValid=cms.vuint32(1),
    iovIsRunNotTime=cms.bool(True),
)
process.cscChannelMapperSource = cms.ESSource(
    'EmptyESSource',
    recordName=cms.string('CSCChannelMapperRecord'),
    firstValid=cms.vuint32(1),
    iovIsRunNotTime=cms.bool(True),
)
process.CSCIndexerESProducer = cms.ESProducer(
    'CSCIndexerESProducer', AlgoName=cms.string('CSCIndexerStartup'))
process.CSCChannelMapperESProducer = cms.ESProducer(
    'CSCChannelMapperESProducer', AlgoName=cms.string('CSCChannelMapperStartup'))

process.unpackTask = cms.Task(process.muonCSCDigis)

# --- ML rechits -> clusters -------------------------------------------------
from HMTntuple.CSCShowerAnalyzer.mlCSCRecHits_cfi import mlCSCRecHits, read_norm_stats
import RecoMuon.MuonRechitClusterProducer.cscRechitClusterProducer_cfi as CSCcluster
from HMTntuple.CSCShowerAnalyzer.custom_mds_cff import cscMDSClusterTable

process.mlCSCRecHits = mlCSCRecHits
if options.statsPath:
    process.mlCSCRecHits.normalization = read_norm_stats(options.statsPath)

process.ca4CSCrechitClustersML = CSCcluster.cscRechitClusterProducer.clone(
    recHitLabel='mlCSCRecHits',
    nRechitMin=50,
    rParam=0.4,
    nStationThres=10,
)

process.mlTask = cms.Task(process.mlCSCRecHits, process.ca4CSCrechitClustersML)

# --- standard rechits -> clusters (reference) -------------------------------
# The same digis also go through the standard CSC local reconstruction, and the
# resulting rechits through an identically configured clusterer. Same events,
# same clustering, so any difference is the ML hits themselves and not the
# downstream treatment.
process.load('RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi')

process.ca4CSCrechitClusters = CSCcluster.cscRechitClusterProducer.clone(
    recHitLabel='csc2DRecHits',
    nRechitMin=50,
    rParam=0.4,
    nStationThres=10,
)

process.refTask = cms.Task(process.csc2DRecHits, process.ca4CSCrechitClusters)

process.p = cms.Path(cms.Sequence(), process.unpackTask, process.mlTask, process.refTask)

# --- output -----------------------------------------------------------------
# Plain EDM output so the ML hits and clusters can be inspected directly. For
# the nano tables, use add_mdsTables(process, MLrechits=True) on top of the
# standard nano config instead -- this file is the minimal RAW driver.
process.out = cms.OutputModule(
    'PoolOutputModule',
    fileName=cms.untracked.string(options.outputFile or 'mlrechits_RAW.root'),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep *_mlCSCRecHits_*_*',
        'keep *_ca4CSCrechitClustersML_*_*',
        'keep *_csc2DRecHits_*_*',
        'keep *_ca4CSCrechitClusters_*_*',
    ),
)
process.end = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.p, process.end)
