# CSC rechit-to-offline-cluster associations

The missing output is an explicit association from saved CSC rechits to their
saved offline cluster rows. The proposed change adds that association to the
existing table producer. It is not a replacement reconstruction or inference
implementation, and it does not change an ONNX model.

This contribution adds two code changes plus this documentation and saved
validation evidence. The code was validated against `asugu/HMTntuple` branch
`ml-csc-rechit-producer` at `01dd35a869b849d2dad978c4e855c318e08f0ee4`, still
the public branch tip when checked on 2026-09-17. The code diff is 25 insertions
and one deletion. It is offered for the maintainer to review, modify or adapt
within the shared reconstruction and inference implementation. Compatibility
with unpublished or subsequent model/inference updates has not been established.

## Exact code changes

| File | Change |
| --- | --- |
| `CSCShowerAnalyzer/plugins/cscMDSshowerTableProducer.cc` | Add signed `ClusterId` to saved constituent rows. Add a configurable extension table containing IDs in original input rechit order. Register the new `rechitTable` string parameter, with an empty default disabling the extension. |
| `CSCShowerAnalyzer/python/custom_mds_cff.py` | When `saveMLrechits` is enabled, attach the nominal ML offline producer's extension to `cscMLRechits`. |

| Branch in the validated configuration | Target row | Unclustered convention |
| --- | --- | --- |
| `cscMDSClusterRechits_ClusterId` | `cscMDSCluster` | Hits omitted from constituent table |
| `cscMDSClusterMLOfflineRechits_ClusterId` | `cscMDSClusterMLOffline` | Hits omitted from constituent table |
| `cscMLRechits_ClusterId` | `cscMDSClusterMLOffline` | `-1` |

IDs are event-local and zero-based after the existing minimum-size cut.
`clsSize.size()` supplies the next saved cluster row; the original rechit
reference key supplies the full-table row. Existing cluster and constituent
ordering is retained. IDs do not associate classical clusters with ML clusters
or refer to the separate HLT-style tables. `IChamber` remains detector metadata.

The producer-level column affects every instance, including optional classical
and ML loose constituent tables when enabled. The full ML extension refers only
to the nominal offline collection. The 100-event fixture used
`looseNRechitMin=0`; extra loose-table branches were not runtime-validated.

## Complete change and dependency accounting

The historical 51-file manifests verify the code-only change: only the two
implementation targets differed from the upstream pin in the tested fixture.
This PR additionally changes the root README and adds this documentation and
evidence. The reconstruction/inference producer, five ONNX files,
normalization JSON, thresholds, reference configuration, and other implementation
files retain their original bytes. There was no model-only substitution in
this validated fixture. This accounting is limited to this fixture and patch,
not every experiment or working copy.

The tested environment also contained a separately supplied
`SimpleMuonRecHitClusterFlatTableProducer` registration and its BuildFile in
`MDSFullprodRuntime/ClusterTable`. Both baseline and patched runs used this
dependency; it is not part of the two-file upstream contribution. The plugin
was copied verbatim from `mds_fullprod` commit
`eb84e40249b8f02612a2f1af50b03a712a0d6dd2`, file
`cmssw/HMTntuple/CSCShowerAnalyzer/plugins/SimpleMuonRecHitClusterFlatTableProducer.cc`.
It registers `SimpleFlatTableProducer<reco::MuonRecHitCluster>`; it does not
implement rechit reconstruction. The supplemental BuildFile needed an explicit
`CommonTools/Utils` dependency after an initial link failure.

The exact dependency manifest is included as
[dependency-manifest.json](evidence/dependency-manifest.json). Its C++ source
SHA-256 is `dd0f7831b07e0dfe11c48b482c21f0f5bbe9b722008166935248932d6676c4ed`;
the BuildFile SHA-256 is
`9ec66e8d6384d3f74af8dec56a32c6d1b59bda6cbf63220c05e69015d8fe847e`.
For reference, the separately supplied implementation was:

```cpp
#include "DataFormats/MuonReco/interface/MuonRecHitCluster.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
typedef SimpleFlatTableProducer<reco::MuonRecHitCluster> SimpleMuonRecHitClusterFlatTableProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleMuonRecHitClusterFlatTableProducer);
```

Its supplemental BuildFile declared:

```xml
<use name="CommonTools/Utils"/>
<use name="DataFormats/MuonReco"/>
<use name="PhysicsTools/NanoAOD"/>
<use name="FWCore/Framework"/>
<library file="*.cc" name="MDSFullprodRuntimeClusterTablePlugins">
  <flags EDM_PLUGIN="1"/>
</library>
```

These snippets document the tested dependency; this contribution does not
install another plugin registration. The maintainer's environment may already
supply it; that has not been established. The contribution does not bootstrap
a missing CMSSW dependency.

## Validation evidence and limits

Historical validation on 2026-09-14 used `CMSSW_13_2_5`,
`el8_amd64_gcc11`, GCC 11.4.1, ROOT 6.26.11 and ONNX Runtime 1.14.1.
`scram b -j 4` and `cmsRun` exited zero. The first 100 events from the exact
input/configuration in [fixture.json](evidence/fixture.json) produced 187
branches: all 184 old branch values, types and titles were unchanged, with
549,008 values compared exactly. All three new branches were signed `Int_t`.

Association checks covered lengths, ID ranges, per-cluster hit counts, row
ordering, centroids and raw/constituent ML intrinsic-field associations.
The sample contained 19 classical clusters / 6,367 constituents and 42 ML
clusters / 7,458 constituents. Of 14,073 full ML hits, 6,615 had ID `-1`.
Zero-cluster and multiple-cluster events occurred on both paths.

- [Historical machine-readable association result](evidence/validation.json)
- [51-file source manifest before](evidence/source-before.json) and
  [after](evidence/source-after.json)

The 2026-09-17 source review checked all 51 before/after hashes against a
fresh upstream checkout, exact two-file code scope, clean patch application,
whitespace and Python syntax. Documentation is additional to that code check. No new CMSSW compilation or event processing was performed for this
review. Historical JSON results were copied unchanged from the saved run
evidence; ROOT comparisons were not rerun today. This is schema-preservation
evidence, not a physics-equivalence result or validation of a newer model.

Before integration, review the following assumptions in the intended version:

- The base ML table and extension must consume the identical rechit collection
  and row order, without independent filtering or sorting.
- The pinned source assumes valid geometry. Its existing FastJet user indices
  use original collection positions while its reference vector skips missing
  chambers; missing geometry could invalidate indexing. The base full-hit
  table also skips missing layers while declaring the original row count.
  The patch preserves these existing behaviors; this sample does not validate
  missing-geometry inputs.
- Loose-table configurations, `saveMLrechits=0`, and updates to inference or
  reconstruction require their own intended-configuration validation.
- Do not install two extensions that provide the same `ClusterId` column to
  `cscMLRechits`. The proposed configuration installs only the nominal one.

The original two-file patch SHA-256 is
`28e3269473cfe5fe90fdc82ec3aad18f6798eb1baee5dda114f12809c232788f`.
This identifies the code-only patch used for the historical test, not the full
PR diff including documentation. Review the intended source revision and
dependency setup before integration; the maintainer can adapt the output
contract to her current shared code.


## Using and checking the association

For an event, constituent row `i` belongs to cluster row `ClusterId[i]` in the
corresponding table listed above. For the full ML table, first require
`ClusterId[i] >= 0`; `-1` is not a valid cluster index. IDs reset each event.
The nominal minimum accepted cluster size in the tested configuration is 50.

The full ML association is enabled through the existing `saveMLrechits=1`
option in `CSCShowerAnalyzer/test/custom_config_RAW_MC_ML_minimal.py`. When
that option is off, the optional full-table extension is not produced; the
constituent ID columns remain part of the producer output. `rechitTable`
is an output table name, not an input collection tag.

To repeat the intended validation in the matching CMSSW environment, use the
exact input, event limit, configuration, models and normalization identified
by [fixture.json](evidence/fixture.json) and the source manifests. Compare
baseline and patched outputs on the same first 100 events. Require all old
branches, types, titles, values and row order to match exactly. Check signed
ID types, column lengths, ID ranges, per-cluster constituent counts, and full
ML associations, including zero-cluster events, multiple-cluster events and
unclustered hits. The saved result does not replace that validation when the
source, model, inference path, input filtering or configuration changes.
