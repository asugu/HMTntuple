#!/bin/bash
# ggH MDSnano production: ML hits + BOTH ML cluster tables (HLT-style and
# offline-style, the latter with its constituent-hit table).
#
# PREREQUISITE -- run this in your own shell first, it prompts for a passphrase:
#     voms-proxy-init -voms cms -valid 24:00
#
# USAGE
#     ./run_ggH_ML.sh              # 5 files, all events   (the default)
#     ./run_ggH_ML.sh 1            # 1 file
#     ./run_ggH_ML.sh 5 1000       # 5 files, 1000 events
#     FILES=a.root,b.root ./run_ggH_ML.sh    # explicit list, skips discovery
#
# The file list is DISCOVERED with `xrdfs ls` rather than hard-coded, because
# the exact 5 files behind mdsnano_ggH_MH125_ML_5files.root were never written
# down. Sorted by the numeric suffix so the choice is reproducible.
set -e

NFILES=${1:-5}
MAXEV=${2:--1}

export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /data/dust/user/guvenlia/cmssw/MDSnano/src
eval `scramv1 runtime -sh`
cd HMTntuple/CSCShowerAnalyzer/test

REDIR=root://cmseos.fnal.gov/
DIR=/store/group/lpclonglived/apresyan/privateProduction/DR/step2_FEVTSIM/Run3_Summer23/ggH_Hto2Sto4B_MH-125-MS-15-ctauS-1000/batch2/v2/ggH_Hto2Sto4B_MH-125-MS-15-ctauS-1000/crab_S23_S2_ggH_Hto2Sto4B_MH-125-MS-15-ctauS-1000_batch2_v2/250805_154619/0000

# The five files behind the mdsnano_ggH_MH125_ML_5files.root BASELINE, per Asu.
# Pinned, not discovered: an `xrdfs ls | sort -V | head -5` picked a DIFFERENT
# five and moved the standard-cluster rate from 0.234 to 0.238 per event, which
# is enough to muddy a before/after on the ML/standard ratio. Keep this list
# unless you deliberately want a different sample.
BASELINE_IDS="999 998 997 996 995"

if [ -z "$FILES" ]; then
  if [ "$NFILES" -le 5 ]; then
    IDS=$(echo $BASELINE_IDS | tr ' ' '\n' | head -n "$NFILES")
  else
    echo ">> more than 5 files asked for; listing $DIR"
    IDS=$(xrdfs $REDIR ls $DIR | grep '\.root$' | sed 's|.*FEVTSIM_||;s|\.root$||' \
          | sort -rn | head -n "$NFILES")
  fi
  FILES=$(for i in $IDS; do echo "$REDIR$DIR/FEVTSIM_$i.root"; done | paste -sd, -)
fi

if [ -z "$FILES" ]; then
  echo "!! no files found -- is the proxy valid? (voms-proxy-info -timeleft)" >&2
  exit 1
fi

echo ">> $(tr ',' '\n' <<<"$FILES" | wc -l) file(s), maxEvents=$MAXEV"
tr ',' '\n' <<<"$FILES" | sed 's|.*/||' | sed 's/^/     /'

# STATS MUST MATCH THE CHECKPOINT THE .onnx GRAPHS CAME FROM.
# The graphs in ../data/ are the v13b_ft_iso 24317153 (event_cluster finetune),
# so the stats that go with them are ../data/norm_stats_SH_event_cluster.json,
# which is the default below and ships alongside the graphs.
# A mismatched stats file corrupts BOTH the input normalization and the output
# denormalization, silently -- the scales differ by more than an order of
# magnitude between trainings (simhits_time std 18.96 here vs 62.25 for
# event_all). Override with
#     STATS=/path/to/norm_stats_SH_other.json ./run_ggH_ML.sh
OUT=mdsnano_ggH_MH125_MLtwin_${NFILES}files.root

cmsRun custom_config_RAW_MC_ML_minimal.py \
  inputFiles="$FILES" \
  statsPath=${STATS:-$CMSSW_BASE/src/HMTntuple/CSCShowerAnalyzer/data/norm_stats_SH_event_cluster.json} \
  saveMLrechits=1 \
  maxEvents=$MAXEV \
  outputFile=$OUT

# NOTE: VarParsing rewrites outputFile when maxEvents is set to a positive
# number -- with maxEvents=1000 the file lands at <name>_numEvent1000.root.
# With maxEvents=-1 the name is left alone.
echo ">> wrote $OUT (check for a _numEventN suffix if you passed maxEvents)"
