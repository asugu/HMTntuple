# MDSnano on signal MC RAW/FEVTSIM: standard + ML clusters, gen, no PAT.
#
# GEN-SIM-RAW -> RAW2DIGI -> CSC local reco -> {csc2DRecHits, mlCSCRecHits}
#   -> clusters -> nano tables, plus a GenPart table carrying decay vertices.
#
# Why not custom_config_RAW_MC_ML.py: that one followed config_raw_mc.py and
# ran Reconstruction_cff -> PATMC_cff -> nanoSequenceMC. The PAT layer dies in
# PATMuonProducer with "ValueMap: no associated value for given product and
# index" on every FEVTSIM sample tried (MH-1000 and MH-125 both, so it is the
# skeleton and not the sample). Nothing in the cluster-efficiency plot needs
# jets, taus, ParticleNet or slimmed muons, so this config does not build them.
# That also matches the "minimalistic workflow" of runCSCShowerAnalyzer_cfg.py.
#
# The MDS side is unchanged: add_mdsTables(MLrechits=True) from custom_mds_cff,
# giving both cluster collections on the same events:
#   cscMDSHLTCluster*   - clusters from standard rechits
#   cscMDSClusterML*    - clusters from ML rechits
#
# Run with e.g.
#   cmsRun custom_config_RAW_MC_ML_minimal.py \
#       inputFiles=root://cmseos.fnal.gov//store/group/... \
#       maxEvents=50 outputFile=mdsnano_sig_ML.root

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register('statsPath',
                 '/data/dust/user/guvenlia/mds-ml/flex_atten/stats/norm_stats_SH_large.json',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'norm_stats_SH_*.json used to train the ONNX graphs')
options.register('saveMLrechits', False,
                 VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 'Also store the ML rechits themselves, not just the clusters')
options.register('looseNRechitMin', 0,
                 VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "If > 0, also emit cscMDSClusterLoose / "
                 "cscMDSClusterMLOfflineLoose with this cluster hit threshold "
                 "instead of the default 50. Used to measure the efficiency "
                 "CEILING -- how many fiducial LLPs left enough energy to "
                 "cluster at all -- separately from the clusterer's own "
                 "efficiency. 0 disables the extra tables entirely.")
options.register('globalTag', '130X_mcRun3_2023_realistic_postBPix_v6',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'Global tag; must match the production campaign')
options.parseArguments()

process = cms.Process('MDSNANO', Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# MLCSCRecHitProducer reports its per-bucket event tally from endStream() as
# LogInfo, which the default threshold drops. limit=-1 lifts the per-category
# occurrence cap. Verbosity only.
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.MLCSCRecHitProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
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
)

# --- RAW -> digis -----------------------------------------------------------
# Only the CSC unpacker: mlCSCRecHits eats wire and strip digis, and the
# standard CSC local reco needs the same digis. Nothing else in this job reads
# tracker/calo digis, so the full RawToDigi sequence is not run.
process.load('EventFilter.CSCRawToDigi.cscUnpacker_cfi')

# ES products the CSC unpacker needs, which the plain RAW workflow does not
# pull in on its own (same boilerplate as runCSCShowerAnalyzer_cfg.py).
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

# --- CSC local reco ---------------------------------------------------------
# Standard rechits. add_mdsTables clusters these into cscMDSHLTCluster and, with
# MLrechits=True, clusters mlCSCRecHits into cscMDSClusterML alongside.
process.load('RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi')

process.recoTask = cms.Task(process.muonCSCDigis, process.csc2DRecHits)

# --- gen --------------------------------------------------------------------
# The stock nano genParticleTable prunes from prunedGenParticles, a PAT product
# this job does not build, so the pruner is re-pointed at the RECO-level
# genParticles that RAW/FEVTSIM carries.
#
# The stock GenPart table also has NO vertex variables -- only kinematics and
# ids. The cluster-efficiency plot is binned in LLP *decay position*, so vx/vy/vz
# are added here. Without them the plot cannot be made at all.
from PhysicsTools.NanoAOD.genparticles_cff import finalGenParticles, genParticleTable
from PhysicsTools.NanoAOD.common_cff import Var

process.finalGenParticles = finalGenParticles.clone(
    src=cms.InputTag('genParticles'),
)
process.genParticleTable = genParticleTable.clone()
# The stock table's only externalVariable is iso = ExtVar(InputTag("genIso")),
# and genIso lives in genParticleTask's PAT-adjacent chain, which this job does
# not run -- it fails at event time with ProductNotFound on edm::ValueMap<float>
# 'genIso'. Nothing here needs gen isolation, so the whole PSet is emptied.
# (Note: iso sits in externalVariables, not variables.)
process.genParticleTable.externalVariables = cms.PSet()
process.genParticleTable.variables.vx = Var('vx', float, precision=10,
                                            doc='production vertex x (cm)')
process.genParticleTable.variables.vy = Var('vy', float, precision=10,
                                            doc='production vertex y (cm)')
process.genParticleTable.variables.vz = Var('vz', float, precision=10,
                                            doc='production vertex z (cm)')

process.genTask = cms.Task(process.finalGenParticles, process.genParticleTable)

# --- path -------------------------------------------------------------------
process.p = cms.Path(cms.Sequence(), process.recoTask, process.genTask)

# --- output -----------------------------------------------------------------
process.NANOAODoutput = cms.OutputModule(
    'NanoAODOutputModule',
    compressionAlgorithm=cms.untracked.string('LZMA'),
    compressionLevel=cms.untracked.int32(9),
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string('NANOAOD'),
        filterName=cms.untracked.string(''),
    ),
    fileName=cms.untracked.string(options.outputFile or 'mdsnano_sig_ML.root'),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep nanoaodFlatTable_*_*_*',
        'keep nanoaodUniqueString_nanoMetadata_*_*',
    ),
)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

process.schedule = cms.Schedule(process.p, process.NANOAODoutput_step)

# --- MDS tables -------------------------------------------------------------
# MLrechits=True is the flag in custom_mds_cff.py, per the "build on
# custom_mds_cff.py" guidance -- not a parallel hand-rolled path.
#
# add_mdsTables ends with process.nanoTableTaskCommon.add(process.MDSTask), and
# that task only exists once the full nano config has been loaded. This job
# deliberately has no nano sequence, so an empty task of that name is created
# first for add_mdsTables to append to; MDSTask is then associated to the path
# explicitly below. Without the stub the call raises AttributeError.
process.nanoTableTaskCommon = cms.Task()

from HMTntuple.CSCShowerAnalyzer.custom_mds_cff import add_mdsTables
process = add_mdsTables(
    process,
    MLrechits=True,
    saveMLrechits=options.saveMLrechits,
    MLstatsPath=options.statsPath or None,
    looseNRechitMin=options.looseNRechitMin,
)

process.p.associate(process.nanoTableTaskCommon)

process.add_(cms.Service('InitRootHandlers', EnableIMT=cms.untracked.bool(False)))
