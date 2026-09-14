// -*- C++ -*-
//
// Package:    HMTntuple/CSCShowerAnalyzer
// Class:      MLCSCRecHitProducer
//
/**\class MLCSCRecHitProducer MLCSCRecHitProducer.cc

  Runs the MDS cross-attention hit-regression network on CSC wire and strip
  digis and emits the predictions as a CSCRecHit2DCollection.

  The point of emitting a *CSCRecHit2DCollection* rather than a bespoke format
  is that RecoMuon/MuonRechitClusterProducer's RechitClusterProducerT is
  hard-typed to that collection but configured by an InputTag. So the existing
  cscRechitClusterProducer and the existing cluster FlatTable work on ML hits
  with nothing but a clone() and a different recHitLabel -- no parallel
  clustering or table infrastructure.

  Workflow (RAW):
      muonCSCDigis  ->  MLCSCRecHitProducer  ->  ca4CSCrechitClustersML
                                             ->  cscMDSClusterTableML

  The network is exported as five fixed-slot "bucket" graphs (4/16/64/256/512
  slots per doc). A "doc" is one CSC chamber-layer. Selection of how many slots
  are real is *inside* the graph (count head's Negative-Binomial upper bound),
  so this producer applies no threshold of its own -- it reads n_keep.

  See onnx/CMSSW_HANDOFF.md in the mds-ml repository for the model contract.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace {

  // ---------------------------------------------------------------------------
  // Feature layout.
  //
  // These orderings are NOT alphabetical and NOT source order. In the training
  // pipeline (mds-ml data_maker.py) the column list was built from a Python
  // *set* union, so the order is whatever that set iteration produced. It was
  // then frozen into the HDF5 metadata ('wire_features' / 'strip_features') and
  // every normalized training file, and therefore into the network weights.
  //
  // The lists below were read back out of stats/test_norm_SH_large.h5 and must
  // match norm_stats_SH_large.json element-for-element. Reordering ANY entry
  // silently feeds the model transposed features -- no crash, just wrong hits.
  // ---------------------------------------------------------------------------

  constexpr int kNWireFeat = 28;
  constexpr int kNStripFeat = 20;

  // Wires: [0..15] TimeBinsOn0..15, [16..18] Start xyz, [19..21] End xyz,
  //        [22..24] Center xyz, [25] Length, [26] WireGroup, [27] DetID
  constexpr int kWTimeBin0 = 0;
  constexpr int kWStartX = 16;
  constexpr int kWEndX = 19;
  constexpr int kWCenterX = 22;
  constexpr int kWLength = 25;
  constexpr int kWWireGroup = 26;
  constexpr int kWDetId = 27;

  // Strips: [0..7] ADC_0..7, [8..10] Start xyz, [11..13] End xyz,
  //         [14..16] Layer xyz, [17] Length, [18] StripNumber, [19] DetID
  constexpr int kSAdc0 = 0;
  constexpr int kSStartX = 8;
  constexpr int kSEndX = 11;
  constexpr int kSLayerX = 14;
  constexpr int kSLength = 17;
  constexpr int kSStripNumber = 18;
  constexpr int kSDetId = 19;

  // Number of ADC time samples the ntupler stored per strip digi.
  constexpr unsigned int kNAdcSamples = 8;

  // The training ntupler wrote at most this many entries of getTimeBinsOn()
  // into a 16-wide window (displacedJetMuon_ntupler.cc: "protect against array
  // overflow", q >= 5 continue). In practice wire digis fire 2-3 time bins, so
  // the cap is essentially never reached -- but it is reproduced here so that a
  // pathological high-multiplicity wire cannot hand the network a pattern that
  // never occurred in training.
  constexpr unsigned int kMaxTimeBinsOn = 5;
  constexpr int kNTimeBinWindow = 16;

  // DetId namespacing used by the graph: it recovers rawId as detid % kBig.
  // This producer runs on the whole event, i.e. a single "cluster", so the
  // cluster index is always 0 and the namespace offset is a no-op. It is kept
  // explicit because the graph's modulo assumes this convention.
  constexpr int64_t kBig = 10000000000LL;

  // Strip selection used in training: a strip is kept when its ADC range
  // exceeds this threshold (data_maker.py STRIP_ADC_THRESHOLD).
  constexpr double kStripAdcRangeThreshold = 13.3;

  // Minmax normalization to [-1, 1], matching
  // H5TabularDataset_SH.normalize_array + normalize_scalar_group:
  //     2 * (x - min) / (max - min + eps) - 1, then clip to [-1, 1]
  // done in float64, so the arithmetic here is double even though the tensor
  // handed to ONNX Runtime is float32.
  constexpr double kNormEps = 1e-8;

  inline double normalizeMinMax(double x, double mn, double mx) {
    const double v = 2.0 * ((x - mn) / (mx - mn + kNormEps)) - 1.0;
    return std::min(1.0, std::max(-1.0, v));
  }

  // --- output denormalization ------------------------------------------------
  //
  // The network was trained on normalized targets and therefore *emits*
  // normalized values. Neither the export nor the ONNX contract check inverts
  // them (the check compared ONNX against torch, and both are normalized), so
  // the inversion is the producer's job. Feeding the raw graph output straight
  // into GlobalPoint puts every hit within ~1 cm of the beamline.
  //
  // Positions use minmax over the simhit position range, so the inverse is the
  // inverse of normalizeMinMax:
  //     x = ((x_norm + 1) / 2) * (max - min) + min
  inline double denormalizeMinMax(double v, double mn, double mx) {
    return ((v + 1.0) / 2.0) * (mx - mn) + mn;
  }

  // Time uses a z-score (stats method "std"), so the inverse is affine.
  inline double denormalizeStd(double z, double mean, double sd) { return z * sd + mean; }

  // Energy is z-scored in LOG10 space: the training target was
  // log10(cscSimHitEloss + log_eps) and then z-scored, so undoing it needs the
  // exponential as well. Without the pow10 the "energy" comes out negative,
  // which is how this bug first showed up.
  inline double denormalizeLogEnergy(double z, double mean, double sd, double logEps) {
    return std::pow(10.0, z * sd + mean) - logEps;
  }

  // --- ME1/1 doc key ---------------------------------------------------------
  //
  // ME1/1 is one physical chamber-layer read out as two strip namespaces:
  // ME1/1b is ring 1 and ME1/a is ring 4. The unpacker forces all ME1/1 *wire*
  // digis to ring 1 (CSCDCCUnpacker.cc: "Allocate all ME1/1 wire digis to ring
  // 1", cfeb=0), while *strips* keep their real icfeb, so ME1/a strips arrive
  // as ring 4. Wires and strips of the same physical layer therefore carry
  // different DetIds and would land in different docs -- and since a doc needs
  // both views, the ring-4 ones get dropped entirely.
  //
  // That they are one plane is measurable, not assumed: in reconstructed
  // rechits the two rings occupy adjacent, non-overlapping bands of the SAME
  // layer-local y (ring 4 spans about [-74, -28] cm, ring 1 about [-34, +72]),
  // and 205 of the station-1 chamber-layers in the test file carry both.
  //
  // So ring 4 is folded onto ring 1 for the purpose of grouping hits into docs.
  // This direction is well defined; the inverse (1 -> {1,4}) is not, and is
  // never needed: the strip's own position features already say where in the
  // plane it sits, so the ring bit is only routing. The emitted rechits keep
  // whatever layer rawId the graph returns.
  inline uint32_t docKeyRawId(const CSCDetId& id) {
    if (id.station() == 1 && id.ring() == 4)
      return CSCDetId(id.endcap(), 1, 1, id.chamber(), id.layer()).rawId();
    return id.rawId();
  }

  // ...and the inverse, applied per emitted hit rather than per doc.
  //
  // Folding ring 4 into ring 1 for grouping means the graph hands back ring-1
  // DetIds for the whole plane, so without this every ME1/a hit would be
  // emitted as ME1/1b -- which shows up as ME1/1 carrying ~2.5x the hits it
  // should while ME1/a stays empty.
  //
  // The two regions are adjacent bands in layer-local y, so the region a hit
  // belongs to follows from its own y. Measured on reconstructed rechits in
  // the test file (station 1, 200 events):
  //
  //     ring 1 (ME1/1b): y in [-34.9, +75.2], 1st pct -32.7
  //     ring 4 (ME1/a) : y in [-74.4, -25.1], 99th pct -30.6
  //
  // The bands overlap slightly, so no cut is exact; y < -31 cm minimises the
  // total misassignment at about 1.4% (1.8% of ring-1 hits and 1.2% of ring-4
  // hits land on the wrong side). x cannot help: ring 4 spans +/-14 cm and
  // ring 1 +/-23 cm, so ring 4 is nested inside ring 1 in x with no separation.
  constexpr double kME1aLocalYMax = -31.0;

  inline CSCDetId splitME11(const CSCDetId& id, double localY) {
    if (id.station() == 1 && id.ring() == 1 && localY < kME1aLocalYMax)
      return CSCDetId(id.endcap(), 1, 4, id.chamber(), id.layer());
    return id;
  }

  // Trapezoid half-width at local y, as used by the training ntupler to derive
  // the wire span. Mirrors CSCLayerGeometry's convention: the chamber widens
  // linearly from the bottom edge to the top edge over its full length.
  inline float trapezoidalWidthAtY(float y, float halfLength, float hbotedge, float htopedge) {
    return 2.0f * (hbotedge + (htopedge - hbotedge) * (y + halfLength) / (2.0f * halfLength));
  }

  // One CSC chamber-layer. Docs are the network's sequence elements.
  struct DocKey {
    uint32_t rawId;
  };

}  // namespace

class MLCSCRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit MLCSCRecHitProducer(const edm::ParameterSet&);
  ~MLCSCRecHitProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // Reports the truncation counters below. Without this they are accumulated
  // and never seen, which would make bucket under-picking exactly the silent
  // failure pickBucket() claims it is not.
  void endStream() override;

  // Builds the per-hit feature rows from digis + geometry.
  void buildWireRows(const CSCWireDigiCollection&,
                     const CSCGeometry&,
                     std::vector<std::array<double, kNWireFeat>>& rows,
                     std::vector<uint32_t>& rawIds) const;

  void buildStripRows(const CSCStripDigiCollection&,
                      const CSCGeometry&,
                      std::vector<std::array<double, kNStripFeat>>& rows,
                      std::vector<uint32_t>& rawIds) const;

  // Chooses which bucket graph to run for a given doc count / occupancy.
  size_t pickBucket(size_t nWires, size_t nStrips) const;

  const edm::EDGetTokenT<CSCWireDigiCollection> wireDigiToken_;
  const edm::EDGetTokenT<CSCStripDigiCollection> stripDigiToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geometryToken_;

  const std::vector<std::string> modelPaths_;
  const std::vector<int> bucketSlots_;
  const double stripAdcRangeThreshold_;
  const bool requireStripWirePair_;
  const unsigned int minFiredStrips_;
  const bool verbose_;

  // Input normalization constants, indexed [feature].
  std::array<double, kNWireFeat> wireMin_, wireMax_;
  std::array<double, kNStripFeat> stripMin_, stripMax_;

  // Output denormalization constants (simhits_pos / _time / _energy).
  // posMin_/posMax_ are (x, y, z); only x and y are used, since the graph does
  // not emit z -- that comes from CSCGeometry.
  std::array<double, 3> posMin_, posMax_;
  double timeMean_ = 0., timeStd_ = 1.;
  double energyMean_ = 0., energyStd_ = 1., energyLogEps_ = 1e-12;

  // One ORT session per bucket. Sessions are per-stream because ORT Run() is
  // thread-safe but we keep the IO buffers as locals anyway.
  Ort::Env env_;
  std::vector<std::unique_ptr<Ort::Session>> sessions_;

  // Counters, reported by endStream(). Not mutable: produce() is non-const,
  // and each stream owns its own producer instance, so these need no locking.
  unsigned long long nEvents_ = 0;
  unsigned long long nDocsSeen_ = 0;
  unsigned long long nDocsTruncated_ = 0;
  unsigned long long nHitsMade_ = 0;
  // How often each bucket graph was chosen. This is the number that says
  // whether the bucket ladder is sized right for production: the event-level
  // max means one busy chamber-layer drags the whole event up, so a ladder
  // that looks fine per-doc can still run every event on the widest graph.
  std::vector<unsigned long long> nEventsPerBucket_;
};

// -----------------------------------------------------------------------------

MLCSCRecHitProducer::MLCSCRecHitProducer(const edm::ParameterSet& iConfig)
    : wireDigiToken_(consumes<CSCWireDigiCollection>(iConfig.getParameter<edm::InputTag>("wireDigiLabel"))),
      stripDigiToken_(consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("stripDigiLabel"))),
      geometryToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()),
      modelPaths_(iConfig.getParameter<std::vector<std::string>>("modelPaths")),
      bucketSlots_(iConfig.getParameter<std::vector<int>>("bucketSlots")),
      stripAdcRangeThreshold_(iConfig.getParameter<double>("stripAdcRangeThreshold")),
      requireStripWirePair_(iConfig.getParameter<bool>("requireStripWirePair")),
      minFiredStrips_(iConfig.getParameter<unsigned int>("minFiredStrips")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      env_(ORT_LOGGING_LEVEL_WARNING, "MLCSCRecHitProducer") {
  if (modelPaths_.size() != bucketSlots_.size()) {
    throw cms::Exception("Configuration")
        << "modelPaths (" << modelPaths_.size() << ") and bucketSlots (" << bucketSlots_.size() << ") must be parallel";
  }
  if (modelPaths_.empty()) {
    throw cms::Exception("Configuration") << "at least one bucket graph is required";
  }
  // Buckets must be ascending so pickBucket() can take the first that fits.
  for (size_t i = 1; i < bucketSlots_.size(); ++i) {
    if (bucketSlots_[i] <= bucketSlots_[i - 1]) {
      throw cms::Exception("Configuration") << "bucketSlots must be strictly ascending";
    }
  }

  // --- normalization stats -------------------------------------------------
  // Read from the ParameterSet rather than parsing norm_stats_SH_large.json
  // here, so the numbers live in the config next to the model paths and a
  // retrain is a config change, not a recompile.
  const auto& normPSet = iConfig.getParameter<edm::ParameterSet>("normalization");
  const auto wMin = normPSet.getParameter<std::vector<double>>("wireMin");
  const auto wMax = normPSet.getParameter<std::vector<double>>("wireMax");
  const auto sMin = normPSet.getParameter<std::vector<double>>("stripMin");
  const auto sMax = normPSet.getParameter<std::vector<double>>("stripMax");

  if (wMin.size() != kNWireFeat || wMax.size() != kNWireFeat) {
    throw cms::Exception("Configuration")
        << "wireMin/wireMax must have " << kNWireFeat << " entries, got " << wMin.size() << "/" << wMax.size();
  }
  if (sMin.size() != kNStripFeat || sMax.size() != kNStripFeat) {
    throw cms::Exception("Configuration")
        << "stripMin/stripMax must have " << kNStripFeat << " entries, got " << sMin.size() << "/" << sMax.size();
  }
  std::copy(wMin.begin(), wMin.end(), wireMin_.begin());
  std::copy(wMax.begin(), wMax.end(), wireMax_.begin());
  std::copy(sMin.begin(), sMin.end(), stripMin_.begin());
  std::copy(sMax.begin(), sMax.end(), stripMax_.begin());

  // Output denormalization: the graph emits normalized x, y, t, E.
  const auto posMin = normPSet.getParameter<std::vector<double>>("posMin");
  const auto posMax = normPSet.getParameter<std::vector<double>>("posMax");
  if (posMin.size() != 3 || posMax.size() != 3) {
    throw cms::Exception("Configuration")
        << "posMin/posMax must have 3 entries (x, y, z), got " << posMin.size() << "/" << posMax.size();
  }
  std::copy(posMin.begin(), posMin.end(), posMin_.begin());
  std::copy(posMax.begin(), posMax.end(), posMax_.begin());

  timeMean_ = normPSet.getParameter<double>("timeMean");
  timeStd_ = normPSet.getParameter<double>("timeStd");
  energyMean_ = normPSet.getParameter<double>("energyMean");
  energyStd_ = normPSet.getParameter<double>("energyStd");
  energyLogEps_ = normPSet.getParameter<double>("energyLogEps");

  // --- ONNX Runtime sessions ----------------------------------------------
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);  // CMSSW owns the threading model
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  for (const auto& p : modelPaths_) {
    const edm::FileInPath fip(p);
    sessions_.emplace_back(std::make_unique<Ort::Session>(env_, fip.fullPath().c_str(), opts));
  }

  nEventsPerBucket_.assign(bucketSlots_.size(), 0);

  produces<CSCRecHit2DCollection>();
}

void MLCSCRecHitProducer::endStream() {
  if (nDocsSeen_ == 0)
    return;

  std::ostringstream buckets;
  for (size_t b = 0; b < bucketSlots_.size(); ++b) {
    const double frac = nEvents_ ? 100.0 * nEventsPerBucket_[b] / nEvents_ : 0.;
    buckets << " s" << bucketSlots_[b] << "=" << nEventsPerBucket_[b] << " (" << std::fixed << std::setprecision(1)
            << frac << "%)";
  }

  const double truncFrac = 100.0 * nDocsTruncated_ / nDocsSeen_;

  edm::LogInfo("MLCSCRecHitProducer")
      << "events=" << nEvents_ << " docs=" << nDocsSeen_ << " hits=" << nHitsMade_ << " truncated docs="
      << nDocsTruncated_ << " (" << std::fixed << std::setprecision(2) << truncFrac << "%)"
      << "\n  bucket chosen per event:" << buckets.str();

  // A doc whose n_keep saturated the bucket lost real hits: the count head
  // asked for more slots than the graph provides. Rare truncation is the
  // expected tail (the NB upper bound is deliberately generous), but a large
  // fraction means the bucket ladder is under-sized for this input and the
  // hit multiplicity is being clipped -- which would look like a physics
  // result rather than a configuration problem, hence the warning.
  if (truncFrac > 1.0) {
    edm::LogWarning("MLCSCRecHitProducer")
        << "n_keep saturated the bucket in " << std::fixed << std::setprecision(2) << truncFrac
        << "% of docs; hits were clipped. Widen bucketSlots or check that the "
           "count head is in-distribution for this input.";
  }
}

void MLCSCRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("wireDigiLabel", edm::InputTag("muonCSCDigis", "MuonCSCWireDigi"))
      ->setComment("unpacked CSC wire digis");
  desc.add<edm::InputTag>("stripDigiLabel", edm::InputTag("muonCSCDigis", "MuonCSCStripDigi"))
      ->setComment("unpacked CSC strip digis");
  desc.add<std::vector<std::string>>("modelPaths", {})->setComment("FileInPath to each bucket ONNX graph, ascending");
  desc.add<std::vector<int>>("bucketSlots", {4, 16, 64, 256, 512})->setComment("slots per doc of each graph");
  desc.add<double>("stripAdcRangeThreshold", kStripAdcRangeThreshold)
      ->setComment("keep strip if max(ADC)-min(ADC) exceeds this (training used 13.3)");
  desc.add<bool>("requireStripWirePair", true)
      ->setComment("keep only chamber-layers that have both wire and strip digis, as in training");
  // 2 since the Sep-2026 rebuild: >=3 was tuned on MinBias for shower purity
  // and drops MIP-like DY muons (~2 fired strips), costing 23 points of DY
  // simhit retention. MUST equal event_datamaker.py's MIN_FIRED_STRIPS.
  desc.add<unsigned int>("minFiredStrips", 2)
      ->setComment(
          "minimum fired strips per chamber-layer; MUST equal event_datamaker.py's "
          "MIN_FIRED_STRIPS or the model sees layer occupancies it never trained on");

  edm::ParameterSetDescription norm;
  norm.add<std::vector<double>>("wireMin", {});
  norm.add<std::vector<double>>("wireMax", {});
  norm.add<std::vector<double>>("stripMin", {});
  norm.add<std::vector<double>>("stripMax", {});
  // Output side: the graph emits normalized x, y, t, E.
  norm.add<std::vector<double>>("posMin", {});
  norm.add<std::vector<double>>("posMax", {});
  norm.add<double>("timeMean", 0.);
  norm.add<double>("timeStd", 1.);
  norm.add<double>("energyMean", 0.);
  norm.add<double>("energyStd", 1.);
  norm.add<double>("energyLogEps", 1e-12);
  desc.add<edm::ParameterSetDescription>("normalization", norm)
      ->setComment("minmax stats from norm_stats_SH_large.json, in frozen feature order");

  desc.addUntracked<bool>("verbose", false);
  descriptions.add("mlCSCRecHits", desc);
}

// -----------------------------------------------------------------------------
// Feature construction.
//
// These two functions reproduce, deliberately line-for-line, the geometry the
// training ntupler computed (cms_lpc_llp/llp_ntupler displacedJetMuon_ntupler.cc,
// the CSC wire and strip digi blocks). The network was trained on those exact
// numbers; deriving "equivalent" quantities a different way would shift the
// inputs off-distribution without any visible error.
// -----------------------------------------------------------------------------

void MLCSCRecHitProducer::buildWireRows(const CSCWireDigiCollection& wireDigis,
                                        const CSCGeometry& geo,
                                        std::vector<std::array<double, kNWireFeat>>& rows,
                                        std::vector<uint32_t>& rawIds) const {
  for (auto detIt = wireDigis.begin(); detIt != wireDigis.end(); ++detIt) {
    const CSCDetId& id = (*detIt).first;
    const CSCLayer* layer = geo.layer(id);
    if (!layer)
      continue;

    const auto* bounds = dynamic_cast<const TrapezoidalPlaneBounds*>(&layer->surface().bounds());
    if (!bounds)
      continue;

    const float halfLength = 0.5f * bounds->length();
    const float htopedge = bounds->width() / 2.0f;
    const float hbotedge = bounds->widthAtHalfLength() - htopedge;
    const GlobalPoint chamberCenter = layer->toGlobal(LocalPoint(0., 0., 0.));

    const auto& range = (*detIt).second;
    for (auto digi = range.first; digi != range.second; ++digi) {
      const int wireGroup = digi->getWireGroup();

      const GlobalPoint wireCenter = layer->centerOfWireGroup(wireGroup);
      const LocalPoint centerLocal = layer->toLocal(wireCenter);

      const float fullWidthAtY = trapezoidalWidthAtY(centerLocal.y(), halfLength, hbotedge, htopedge);
      const float halfWireSpan = 0.5f * fullWidthAtY;

      // Wire direction is tangential, i.e. perpendicular in 2D to the radial
      // direction from the chamber centre to the wire centre.
      const GlobalVector radialDir = (wireCenter - chamberCenter).unit();
      const GlobalVector wireDir(-radialDir.y(), radialDir.x(), 0.);

      const GlobalPoint wireStart = wireCenter - wireDir * halfWireSpan;
      const GlobalPoint wireEnd = wireCenter + wireDir * halfWireSpan;
      const double wireLength = (wireEnd - wireStart).mag();

      std::array<double, kNWireFeat> row{};
      row.fill(0.);

      // Time bins: a 16-wide occupancy window, filled from the first
      // kMaxTimeBinsOn entries of getTimeBinsOn(). Stored values are 1-based,
      // so bin value v sets column v-1.
      const std::vector<int> tbins = digi->getTimeBinsOn();
      const unsigned int nTb = std::min<unsigned int>(tbins.size(), kMaxTimeBinsOn);
      for (unsigned int q = 0; q < nTb; ++q) {
        const int v = tbins[q];
        if (v >= 1 && v <= kNTimeBinWindow)
          row[kWTimeBin0 + (v - 1)] = 1.;
      }

      row[kWStartX + 0] = wireStart.x();
      row[kWStartX + 1] = wireStart.y();
      row[kWStartX + 2] = wireStart.z();
      row[kWEndX + 0] = wireEnd.x();
      row[kWEndX + 1] = wireEnd.y();
      row[kWEndX + 2] = wireEnd.z();
      row[kWCenterX + 0] = wireCenter.x();
      row[kWCenterX + 1] = wireCenter.y();
      row[kWCenterX + 2] = wireCenter.z();
      row[kWLength] = wireLength;
      row[kWWireGroup] = wireGroup;
      row[kWDetId] = static_cast<double>(id.rawId());

      rows.push_back(row);
      // Grouping key, not the true id: see docKeyRawId. Wires are already
      // ring 1 in ME1/1, so this is a no-op for them -- applied anyway so
      // both views go through exactly the same keying.
      rawIds.push_back(docKeyRawId(id));
    }
  }
}

void MLCSCRecHitProducer::buildStripRows(const CSCStripDigiCollection& stripDigis,
                                         const CSCGeometry& geo,
                                         std::vector<std::array<double, kNStripFeat>>& rows,
                                         std::vector<uint32_t>& rawIds) const {
  for (auto detIt = stripDigis.begin(); detIt != stripDigis.end(); ++detIt) {
    const CSCDetId& id = (*detIt).first;
    const CSCLayer* layer = geo.layer(id);
    if (!layer)
      continue;

    const auto* bounds = dynamic_cast<const TrapezoidalPlaneBounds*>(&layer->surface().bounds());
    if (!bounds)
      continue;
    const float halfLength = 0.5f * bounds->length();

    const auto& range = (*detIt).second;
    for (auto digi = range.first; digi != range.second; ++digi) {
      const std::vector<int>& adc = digi->getADCCounts();
      if (adc.size() < kNAdcSamples)
        continue;

      // Training selection: keep the strip only if its ADC swing is large
      // enough. Applied here on the same 8 samples the features carry.
      const auto mm = std::minmax_element(adc.begin(), adc.begin() + kNAdcSamples);
      const double adcRange = static_cast<double>(*mm.second) - static_cast<double>(*mm.first);
      if (adcRange <= stripAdcRangeThreshold_)
        continue;

      const int stripNum = digi->getStrip();
      const GlobalPoint stripCenter = layer->centerOfStrip(stripNum);
      const LocalPoint centerLocal = layer->toLocal(stripCenter);

      // Strip endpoints are the strip centre extended to the chamber edges
      // along local y (the strip runs radially, unlike the wire).
      const LocalPoint localStart(centerLocal.x(), -halfLength, centerLocal.z());
      const LocalPoint localEnd(centerLocal.x(), +halfLength, centerLocal.z());
      const GlobalPoint stripStart = layer->toGlobal(localStart);
      const GlobalPoint stripEnd = layer->toGlobal(localEnd);
      const double stripLength = (stripEnd - stripStart).mag();

      std::array<double, kNStripFeat> row{};
      row.fill(0.);

      for (unsigned int i = 0; i < kNAdcSamples; ++i)
        row[kSAdc0 + i] = static_cast<double>(adc[i]);

      row[kSStartX + 0] = stripStart.x();
      row[kSStartX + 1] = stripStart.y();
      row[kSStartX + 2] = stripStart.z();
      row[kSEndX + 0] = stripEnd.x();
      row[kSEndX + 1] = stripEnd.y();
      row[kSEndX + 2] = stripEnd.z();
      row[kSLayerX + 0] = stripCenter.x();
      row[kSLayerX + 1] = stripCenter.y();
      row[kSLayerX + 2] = stripCenter.z();
      row[kSLength] = stripLength;
      row[kSStripNumber] = stripNum;
      row[kSDetId] = static_cast<double>(id.rawId());

      rows.push_back(row);
      // ME1/a strips (ring 4) fold onto ring 1 here so they share a doc with
      // the ME1/1 wires of the same physical layer; without this they have no
      // wire partner and the whole chamber-layer is dropped.
      rawIds.push_back(docKeyRawId(id));
    }
  }
}

size_t MLCSCRecHitProducer::pickBucket(size_t nWires, size_t nStrips) const {
  // The count head decides how many hits a doc gets, but we must choose the
  // graph *before* running it. Digi occupancy is the only cheap proxy
  // available: use the larger of the two per-doc digi counts as a lower bound
  // on the hit multiplicity and take the smallest bucket that covers it.
  //
  // Under-picking is visible (n_keep saturates at the bucket size and the
  // truncation counter fires), so this is a tunable, not a silent failure.
  const size_t occupancy = std::max(nWires, nStrips);
  for (size_t b = 0; b < bucketSlots_.size(); ++b) {
    if (occupancy <= static_cast<size_t>(bucketSlots_[b]))
      return b;
  }
  return bucketSlots_.size() - 1;
}

// -----------------------------------------------------------------------------

void MLCSCRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geo = iSetup.getData(geometryToken_);
  const auto& wireDigis = iEvent.get(wireDigiToken_);
  const auto& stripDigis = iEvent.get(stripDigiToken_);

  // Counted before the early returns below, so the per-bucket fractions in
  // endStream() are over all events, not only the ones that reached inference.
  nEvents_++;

  auto out = std::make_unique<CSCRecHit2DCollection>();

  std::vector<std::array<double, kNWireFeat>> wireRows;
  std::vector<uint32_t> wireIds;
  buildWireRows(wireDigis, geo, wireRows, wireIds);

  std::vector<std::array<double, kNStripFeat>> stripRows;
  std::vector<uint32_t> stripIds;
  buildStripRows(stripDigis, geo, stripRows, stripIds);

  if (wireRows.empty() || stripRows.empty()) {
    iEvent.put(std::move(out));
    return;
  }

  // --- docs ----------------------------------------------------------------
  // A doc is a chamber-layer. Training built docs from chamber-layers that had
  // both wire and strip content, so by default we do the same: a doc with only
  // one view is a configuration the decoder never saw.
  std::map<uint32_t, size_t> docIndex;
  {
    std::map<uint32_t, std::pair<bool, bool>> seen;  // rawId -> (hasWire, hasStrip)
    std::map<uint32_t, unsigned int> nFiredStrips;   // rawId -> fired strip count
    for (auto id : wireIds)
      seen[id].first = true;
    for (auto id : stripIds) {
      seen[id].second = true;
      nFiredStrips[id]++;
    }
    for (const auto& kv : seen) {
      if (requireStripWirePair_ && !(kv.second.first && kv.second.second))
        continue;
      // Minimum fired-strip occupancy per chamber-layer.
      //
      // This MUST match event_datamaker.py's MIN_FIRED_STRIPS. The training
      // docs are built with the same rule, and a layer configuration the model
      // never saw is silently out-of-distribution rather than an error.
      //
      // "Fired" means the same thing on both sides: buildStripRows() already
      // applied the max(ADC)-min(ADC) > stripAdcRangeThreshold_ cut, which is
      // data_maker's STRIP_ADC_THRESHOLD. So the two thresholds must agree
      // too -- both default to 13.3.
      //
      // Chosen as 3 from a scan on MinBias (400 events): relative to requiring
      // a reconstructed rechit (what the training used to do), >=3 fired
      // strips keeps 3.2x as many layers that carry a real energy deposit,
      // at higher purity and a LOWER empty-layer fraction.
      if (minFiredStrips_ > 1) {
        auto it = nFiredStrips.find(kv.first);
        if (it == nFiredStrips.end() || it->second < minFiredStrips_)
          continue;
      }
      const size_t next = docIndex.size();
      docIndex.emplace(kv.first, next);
    }
  }
  const size_t nDocs = docIndex.size();
  if (nDocs == 0) {
    iEvent.put(std::move(out));
    return;
  }

  // --- assemble tensors ----------------------------------------------------
  // Rows whose chamber-layer did not become a doc are dropped here, and their
  // feature row must be dropped with them -- an index without its row silently
  // misaligns every hit after it.
  std::vector<float> wireFeat, stripFeat;
  std::vector<int64_t> docIdxWires, docIdxStrips, detIdxWires, detIdxStrips;
  std::vector<size_t> perDocWires(nDocs, 0), perDocStrips(nDocs, 0);

  wireFeat.reserve(wireRows.size() * kNWireFeat);
  for (size_t i = 0; i < wireRows.size(); ++i) {
    auto it = docIndex.find(wireIds[i]);
    if (it == docIndex.end())
      continue;
    for (int f = 0; f < kNWireFeat; ++f)
      wireFeat.push_back(static_cast<float>(normalizeMinMax(wireRows[i][f], wireMin_[f], wireMax_[f])));
    docIdxWires.push_back(static_cast<int64_t>(it->second));
    detIdxWires.push_back(static_cast<int64_t>(wireIds[i]));  // cluster index 0 -> no kBig offset
    perDocWires[it->second]++;
  }

  stripFeat.reserve(stripRows.size() * kNStripFeat);
  for (size_t i = 0; i < stripRows.size(); ++i) {
    auto it = docIndex.find(stripIds[i]);
    if (it == docIndex.end())
      continue;
    for (int f = 0; f < kNStripFeat; ++f)
      stripFeat.push_back(static_cast<float>(normalizeMinMax(stripRows[i][f], stripMin_[f], stripMax_[f])));
    docIdxStrips.push_back(static_cast<int64_t>(it->second));
    detIdxStrips.push_back(static_cast<int64_t>(stripIds[i]));
    perDocStrips[it->second]++;
  }

  if (docIdxWires.empty() || docIdxStrips.empty()) {
    iEvent.put(std::move(out));
    return;
  }

  std::vector<int64_t> detIdxDocs(nDocs, 0);
  std::vector<uint32_t> docRawId(nDocs, 0);
  for (const auto& kv : docIndex) {
    detIdxDocs[kv.second] = static_cast<int64_t>(kv.first);
    docRawId[kv.second] = kv.first;
  }

  // One graph runs the whole event, so the bucket must cover the busiest doc.
  //
  // This is the event-level MAX, which is conservative and it is the dominant
  // cost of this producer: decoder cost is linear in the slot count, so a
  // single busy chamber-layer pushes all of the event's docs onto the widest
  // graph. Measured slot demand is heavily skewed (p50=5, p90=33), so most
  // docs run a vastly oversized attention.
  //
  // WHY NOT PER-DOC BUCKETING (the obvious fix -- do not "just" implement it):
  // it would mean partitioning the event's docs into groups and running each
  // group as its own call. Slot outputs ARE independent of the slot count
  // (decoder slots attend only to wires, never to each other -- see
  // onnx/BUCKETING_HANDOFF.md), so splitting SLOTS is safe. But splitting
  // DOCS is not, because two things in the model couple docs to each other:
  //
  //   1. the encoder mask is "adjacent" (|q_det - k_det| <= 1 on the raw
  //      DetId), so a wire attends to wires in NEIGHBOURING layers. Grouping
  //      docs by bucket scatters neighbouring layers into different calls and
  //      silently drops those cross-layer edges.
  //   2. the count head's cluster-level features (sum_w_doc, sum_s_doc,
  //      n_layers_doc; cross_attention_flex.py:889) aggregate over every doc
  //      in the call. Per-group calls turn them into per-group totals, which
  //      changes n_keep -- i.e. changes the hit multiplicity that is already
  //      the open issue with this model.
  //
  // Neither shows up as an error; both change the answers. The natural place
  // to fix this is the count-head retrain that is already planned for the
  // multiplicity problem: cardinality features computed over a LOCAL
  // neighbourhood rather than the whole call would make doc groups genuinely
  // separable, at which point per-doc bucketing becomes safe and cheap.
  size_t bucket = 0;
  for (size_t d = 0; d < nDocs; ++d)
    bucket = std::max(bucket, pickBucket(perDocWires[d], perDocStrips[d]));
  const int nSlots = bucketSlots_[bucket];
  nEventsPerBucket_[bucket]++;

  // --- run -----------------------------------------------------------------
  auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const int64_t nW = static_cast<int64_t>(docIdxWires.size());
  const int64_t nS = static_cast<int64_t>(docIdxStrips.size());
  const int64_t nD = static_cast<int64_t>(nDocs);

  const std::array<int64_t, 2> shapeW{{nW, kNWireFeat}};
  const std::array<int64_t, 2> shapeS{{nS, kNStripFeat}};
  const std::array<int64_t, 1> shapeIdxW{{nW}};
  const std::array<int64_t, 1> shapeIdxS{{nS}};
  const std::array<int64_t, 1> shapeD{{nD}};

  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(memInfo, wireFeat.data(), wireFeat.size(), shapeW.data(), 2));
  inputs.push_back(Ort::Value::CreateTensor<float>(memInfo, stripFeat.data(), stripFeat.size(), shapeS.data(), 2));
  inputs.push_back(
      Ort::Value::CreateTensor<int64_t>(memInfo, docIdxWires.data(), docIdxWires.size(), shapeIdxW.data(), 1));
  inputs.push_back(
      Ort::Value::CreateTensor<int64_t>(memInfo, docIdxStrips.data(), docIdxStrips.size(), shapeIdxS.data(), 1));
  inputs.push_back(
      Ort::Value::CreateTensor<int64_t>(memInfo, detIdxWires.data(), detIdxWires.size(), shapeIdxW.data(), 1));
  inputs.push_back(
      Ort::Value::CreateTensor<int64_t>(memInfo, detIdxStrips.data(), detIdxStrips.size(), shapeIdxS.data(), 1));
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(memInfo, detIdxDocs.data(), detIdxDocs.size(), shapeD.data(), 1));

  static const char* kInputNames[] = {
      "wires", "strips", "doc_idx_wires", "doc_idx_strips", "det_idx_wires", "det_idx_strips", "det_idx_docs"};
  static const char* kOutputNames[] = {"hits", "hit_detid", "n_keep", "count_mu"};

  auto outputs = sessions_[bucket]->Run(
      Ort::RunOptions{nullptr}, kInputNames, inputs.data(), inputs.size(), kOutputNames, 4);

  const float* hits = outputs[0].GetTensorData<float>();          // (nDocs, nSlots, 4) x,y,t,E
  const int64_t* hitDetId = outputs[1].GetTensorData<int64_t>();  // (nDocs, nSlots)
  const int32_t* nKeep = outputs[2].GetTensorData<int32_t>();     // (nDocs,)

  // --- convert to CSCRecHit2D ----------------------------------------------
  // Local positions are stored in the LAYER frame -- the CSC convention, and
  // what a real csc2DRecHit carries. See the longer note at the toLocal() call
  // below for why that is right even though the cluster producer converts back
  // with geo.chamber(), which is not its inverse.
  std::vector<std::pair<CSCDetId, CSCRecHit2D>> made;
  made.reserve(nDocs * 4);

  for (size_t d = 0; d < nDocs; ++d) {
    const int keep = std::min(nKeep[d], nSlots);
    nDocsSeen_++;
    if (keep >= nSlots)
      nDocsTruncated_++;

    for (int s = 0; s < keep; ++s) {
      const size_t base = (d * static_cast<size_t>(nSlots) + static_cast<size_t>(s)) * 4;

      // The graph emits NORMALIZED x, y, t, E -- invert each with the same
      // stats the training used, otherwise every hit lands within ~1 cm of the
      // beamline and the "energy" comes out negative.
      const double gx = denormalizeMinMax(hits[base + 0], posMin_[0], posMax_[0]);
      const double gy = denormalizeMinMax(hits[base + 1], posMin_[1], posMax_[1]);
      const double tt = denormalizeStd(hits[base + 2], timeMean_, timeStd_);
      const double ee = denormalizeLogEnergy(hits[base + 3], energyMean_, energyStd_, energyLogEps_);

      // The graph emits the layer rawId per slot; docRawId is the fallback for
      // the (not expected) case where the slot carries no id of its own.
      const int64_t rawSlot = hitDetId[d * static_cast<size_t>(nSlots) + static_cast<size_t>(s)] % kBig;
      const uint32_t rawId = rawSlot > 0 ? static_cast<uint32_t>(rawSlot) : docRawId[d];

      const CSCDetId layerId(rawId);
      const CSCLayer* layer = geo.layer(layerId);
      if (!layer)
        continue;

      // z is not predicted by the network -- it is the layer's own plane, which
      // is exactly where a hit in this layer must lie.
      const double zLayer = layer->toGlobal(LocalPoint(0., 0., 0.)).z();

      // Store the local position in the LAYER frame, which is the CSC
      // convention and what a real csc2DRecHit carries.
      //
      // Note this is deliberately NOT the inverse of what the cluster producer
      // does: RechitClusterProducer converts back with
      // geo.chamber(detid)->toGlobal(), so a layer-frame point picks up the
      // layer's small z offset within the chamber (~1.7 cm). Matching the
      // rechit convention is the right call anyway -- it means ML hits and real
      // rechits are treated identically by everything downstream, including
      // that quirk, so the two are directly comparable.
      const LocalPoint localPos = layer->toLocal(GlobalPoint(gx, gy, zLayer));

      // Recover the ME1/1b / ME1/a split that docKeyRawId collapsed. The local
      // position is already in the layer frame, which is the frame the y cut
      // was measured in. ME1/1b and ME1/a share the plane, so the geometry and
      // hence localPos are unaffected -- only the DetId label changes.
      const CSCDetId hitId = splitME11(layerId, localPos.y());

      // Dummy uncertainty for now; a real LocalError is a follow-up.
      const LocalError localErr(1.f, 0.f, 1.f);

      // These are placeholders: the ML hit has no strip/wire-group identity to
      // report. Nothing downstream in the clusterer reads them -- it uses only
      // localPosition(), cscDetId(), tpeak() and wireTime().
      //
      // They are deliberately pushed to ZERO wherever the data format allows,
      // so that a consumer can tell an ML hit from a real one. Real rechits
      // have hitWire >= 1 and nWireGroups >= 1 (wire numbering is 1-based;
      // measured, 0 of 37273 real cluster hits carry a zero), so a zero here is
      // unambiguous rather than merely unusual.
      //
      // wgroups EMPTY is legal and explicitly supported: CSCRecHit2D.cc guards
      // it (`if (nWireGroups_ > 0) {...} else { hitWire_ = 0; theWGroupsBX_ = 0; }`),
      // giving nWireGroups() == 0, hitWire() == 0, wgroupsBX() == 0.
      //
      // channels EMPTY is NOT legal and must stay a one-element dummy. The
      // constructor ends with `nTimeBins_ = adcs.size() / nStrips_` where
      // nStrips_ = channels.size(), so an empty container is an integer divide
      // by zero -- SIGFPE, not an exception. nStrips() is therefore stuck at 1
      // and is the ONE placeholder that cannot be zeroed. Do not "clean it up"
      // to match the others; it crashes. Note 0.85% of real hits also have
      // nStrips == 1, so nStrips is not a usable ML tag either way -- use
      // hitWire == 0.
      //
      // The ADC container stays empty, which makes nTimeBins_ zero and leaves
      // theADCs_ at its zero-initialized state.
      CSCRecHit2D::ChannelContainer channels{0};
      CSCRecHit2D::ADCContainer adcs;
      CSCRecHit2D::ChannelContainer wgroups;

      made.emplace_back(hitId,
                        CSCRecHit2D(hitId,
                                    localPos,
                                    localErr,
                                    channels,
                                    adcs,
                                    wgroups,
                                    static_cast<float>(tt),  // tpeak
                                    0.f,    // positionWithinStrip
                                    0.f,    // errorWithinStrip
                                    0,      // quality
                                    0,      // badStrip
                                    0,      // badWireGroup
                                    static_cast<int>(std::lround(tt * 100.0)),  // wireTime is stored scaled by 100
                                    static_cast<float>(ee)));  // energyDepositedInLayer
      nHitsMade_++;
    }
  }

  // CSCRecHit2DCollection is a RangeMap: hits must be inserted grouped by DetId.
  std::sort(made.begin(), made.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (auto it = made.begin(); it != made.end();) {
    auto jt = it;
    while (jt != made.end() && jt->first == it->first)
      ++jt;
    std::vector<CSCRecHit2D> group;
    group.reserve(std::distance(it, jt));
    for (auto kt = it; kt != jt; ++kt)
      group.push_back(kt->second);
    out->put(it->first, group.begin(), group.end());
    it = jt;
  }

  if (verbose_) {
    edm::LogInfo("MLCSCRecHitProducer")
        << "docs=" << nDocs << " bucket=" << nSlots << " wires=" << nW << " strips=" << nS << " hits=" << made.size();
  }

  iEvent.put(std::move(out));
}

DEFINE_FWK_MODULE(MLCSCRecHitProducer);
