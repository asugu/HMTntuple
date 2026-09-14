# MDSnano on signal MC RAW (FEVTSIM) with both standard and ML rechit clusters.
#
# GEN-SIM-RAW -> RAW2DIGI -> CSC local reco -> {standard, ML} rechits ->
# clusters -> nano tables, with the full MC nano sequence so GenPart is there.
#
# This is custom_config_AOD_mc.py (nanoSequenceMC + add_mdsTables) with the
# RAW->RECO front end from custom_config_RAW_ML.py bolted on, because the ML
# producer eats wire/strip digis and those do not exist on AOD. The ML side is
# add_mdsTables(MLrechits=True) -- the flag in custom_mds_cff.py -- not a
# parallel hand-rolled path, per the "build on custom_mds_cff.py" guidance.
#
# Output carries both cluster collections on the same events:
#   cscMDSHLTCluster*    - clusters from standard csc2DRecHits
#   cscMDSClusterML*     - clusters from mlCSCRecHits
# plus GenPart, which is what the cluster-efficiency-vs-decay-position plot
# needs for its denominator.
#
# Run with e.g.
#   cmsRun custom_config_RAW_MC_ML.py \
#       inputFiles=root://cmseos.fnal.gov//store/group/... \
#       statsPath=/data/dust/user/guvenlia/mds-ml/flex_atten/stats/norm_stats_SH_large.json \
#       globalTag=auto:phase1_2023_realistic_postBPix \
#       maxEvents=50 outputFile=mdsnano_sig_ML.root

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register('statsPath',
                 '/data/dust/user/guvenlia/mds-ml/flex_atten/stats/norm_stats_SH_large.json',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'norm_stats_SH_*.json used to train the ONNX graphs. Required in '
                 'practice: without it the normalization PSet is empty and the '
                 'producer throws.')
options.register('saveMLrechits', False,
                 VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 'Also store the ML rechits themselves, not just the clusters')
# Run3 Summer23 signal. Must match the campaign the sample was produced with,
# or the CSC digi unpacking is wrong.
options.register('globalTag', 'auto:phase1_2023_realistic_postBPix',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'Global tag')
options.parseArguments()

process = cms.Process('MDSNANO', Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# --- standard configurations ------------------------------------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.source = cms.Source(
    'PoolSource',
    fileNames=cms.untracked.vstring(options.inputFiles),
    secondaryFileNames=cms.untracked.vstring(),
)
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.maxEvents),
    output=cms.optional.untracked.allowed(cms.int32, cms.PSet),
)

process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(0),
    wantSummary=cms.untracked.bool(False),
)

# --- CSC unpacker ES boilerplate --------------------------------------------
# The CSC unpacker needs these two ES products, which the plain RAW workflow
# does not pull in on its own (same boilerplate as runCSCShowerAnalyzer_cfg.py
# and custom_config_RAW_ML.py).
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

# --- steps ------------------------------------------------------------------
# Full RAW2DIGI + local reco: PAT/nano needs far more than the CSC chain
# (tracks, vertices, PF), so the standard sequences run rather than just the
# CSC unpacker. muonCSCDigis comes along inside RawToDigi, which is what
# mlCSCRecHits consumes.
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.nanoAOD_step = cms.Path(process.nanoSequenceMC)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.NANOAODoutput = cms.OutputModule(
    'NanoAODOutputModule',
    compressionAlgorithm=cms.untracked.string('LZMA'),
    compressionLevel=cms.untracked.int32(9),
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string('NANOAOD'),
        filterName=cms.untracked.string(''),
    ),
    fileName=cms.untracked.string(options.outputFile or 'mdsnano_sig_ML.root'),
    outputCommands=process.NANOAODEventContent.outputCommands,
)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

process.schedule = cms.Schedule(
    process.raw2digi_step,
    process.reconstruction_step,
    process.nanoAOD_step,
    process.endjob_step,
    process.NANOAODoutput_step,
)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# --- customisation ----------------------------------------------------------
from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon
process = nanoAOD_customizeCommon(process)

from HMTntuple.CSCShowerAnalyzer.custom_mds_cff import add_mdsTables
process = add_mdsTables(
    process,
    MLrechits=True,
    saveMLrechits=options.saveMLrechits,
    MLstatsPath=options.statsPath or None,
)

from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC
process = miniAOD_customizeAllMC(process)

process.add_(cms.Service('InitRootHandlers', EnableIMT=cms.untracked.bool(False)))

from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
