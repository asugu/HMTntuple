// -*- C++ -*-
//
// Package:    HMTntuple/CSCShowerAnalyzer
// Class:      MLCSCRecHitProducer
//
/**\class MLCSCRecHitProducer MLCSCRecHitProducer.cc

  Runs the classical-mix CSC hit-regression network (cm_csc_v*) on CSC wire and
  strip digis and emits the predictions as a CSCRecHit2DCollection, so the stock
  cscRechitClusterProducer and the cluster tables run on it with a clone().

  Workflow (RAW):
      muonCSCDigis  ->  MLCSCRecHitProducer  ->  ca4CSCrechitClustersML / mlMuonHitClusters

  Two ONNX graphs (encoder+count, decoder), export_ragged_cmssw.py. A doc is one
  CSC layer. Inputs reproduce mds-ml/flex_atten csc_event_datamaker.py (docs, ME1/1
  wire copies, local (u,v)) + classicalmix_dataset.py __getitem__ (normalisation);
  outputs are inverted as CSCClassicalMixDataset.denormalize_target.
  - local (u,v) per layer: v along the layer's local +y projected on global xy,
    u = that axis turned by +90 deg (packed_io.csc_to_local); both in cm
  - ME1/1 unfolded: ME1/a and ME1/1b are separate docs; ME1/1 wire digis (all
    under ME1/1b) are copied into each ME1/1 doc whose strip v-range their band crosses
  - no min-fired-strips cut; graph B emits normalised (u_n, v_n, t, E)
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
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
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace {

  // classicalmix_dataset.py: wires = 16 time bins | wire group, start u,v, end u,v, width, centre x,y,z
  //                          strips = 8 ADC     | strip number, start u,v, end u,v, centre x,y,z
  constexpr int kNTimeBins = 16;
  constexpr unsigned int kNAdc = 8;
  constexpr unsigned int kMaxTimeBinsOn = 5;  // the ntupler stores the first 5 getTimeBinsOn() entries
  constexpr int kNWireFeat = kNTimeBins + 9;
  constexpr int kNStripFeat = kNAdc + 8;
  constexpr int64_t kBig = 10000000000LL;  // graph namespace; one event = cluster 0

  // classicalmix_dataset._minmax: 2(x-min)/(max-min+1e-12) - 1, NOT clipped
  inline double minmax(double x, double mn, double mx) { return 2.0 * (x - mn) / (mx - mn + 1e-12) - 1.0; }

  // raw DetId bits as in csc_event_datamaker.py (station 1 raw ring: ME1/a 1, ME1/1b 2, ME1/2 3, ME1/3 4)
  inline int rawStation(uint32_t r) { return (r >> 12) & 7; }
  inline int rawRing(uint32_t r) { return (r >> 9) & 7; }
  inline bool isME11(uint32_t r) { return rawStation(r) == 1 && (rawRing(r) == 1 || rawRing(r) == 2); }
  inline uint32_t toME11b(uint32_t r) { return (r & ~(7u << 9)) | (2u << 9); }

  // packed_io.csc_frames(): layer centre and the local +y axis in global xy
  struct Frame {
    double cx = 0., cy = 0., cz = 0., c = 1., s = 0.;
    void toLocal(double x, double y, double& u, double& v) const {
      const double dx = x - cx, dy = y - cy;
      u = -s * dx + c * dy;
      v = c * dx + s * dy;
    }
    void toGlobal(double u, double v, double& x, double& y) const {
      x = cx - s * u + c * v;
      y = cy + c * u + s * v;
    }
  };

  inline Frame frameOf(const CSCLayer* layer) {
    Frame f;
    const GlobalPoint o = layer->toGlobal(LocalPoint(0., 0., 0.));
    const GlobalVector yh = layer->toGlobal(LocalVector(0., 1., 0.));
    const double n = std::hypot(yh.x(), yh.y());
    f.cx = o.x();
    f.cy = o.y();
    f.cz = o.z();
    f.c = n > 0. ? yh.x() / n : 1.;
    f.s = n > 0. ? yh.y() / n : 0.;
    return f;
  }

  // classicalmix stats "trapezoid" per (station, raw ring): v -> [-1,1] over the strip plane, u / half-width at v
  struct Trap {
    double vlo = -1., vhi = 1., hwlo = 1., hwhi = 1.;
    void norm(double u, double v, double& un, double& vn) const {
      vn = 2.0 * (v - vlo) / (vhi - vlo) - 1.0;
      un = u / (hwlo + (hwhi - hwlo) * (vn + 1.0) / 2.0);
    }
    void denorm(double un, double vn, double& u, double& v) const {
      u = un * (hwlo + (hwhi - hwlo) * (vn + 1.0) / 2.0);
      v = vlo + (vn + 1.0) / 2.0 * (vhi - vlo);
    }
  };

  struct StripRow {
    uint32_t raw;
    int strip;
    std::array<double, kNAdc> adc;
    double u0, v0, u1, v1;
  };
  struct WireRow {
    uint32_t raw;  // as unpacked: ME1/1 wires are ME1/1b
    int group;
    std::array<double, kNTimeBins> tb;
    double gx0, gy0, gx1, gy1;  // global ends (float32, as the ntupler stores them)
    double u0, v0, u1, v1;      // in the wire's own layer frame
    double width;
  };

  // csc_event_datamaker.wire_band_v: v-extent of the middle wire +- width/2 across it
  inline std::pair<double, double> wireBandV(const WireRow& w) {
    const double len = std::max(std::hypot(w.u1 - w.u0, w.v1 - w.v0), 1e-6);
    const double half = 0.5 * w.width * std::abs(w.u1 - w.u0) / len;
    return {std::min(w.v0, w.v1) - half, std::max(w.v0, w.v1) + half};
  }

}  // namespace

class MLCSCRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit MLCSCRecHitProducer(const edm::ParameterSet&);
  ~MLCSCRecHitProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  const Trap& trapOf(uint32_t raw) const;

  const edm::EDGetTokenT<CSCWireDigiCollection> wireDigiToken_;
  const edm::EDGetTokenT<CSCStripDigiCollection> stripDigiToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geometryToken_;

  const std::string encoderPath_;
  const std::string decoderPath_;
  const std::string nKeepRule_;
  const double slotZ_;
  const double stripAdcRangeThreshold_;
  const int minKeep_;
  const double minKeepMinMu_;
  const bool verbose_;
  bool stripsToDecoder_ = false;
  bool decoderHasSigma_ = false;

  // input normalisation (classicalmix stats "minmax", "layer_centre", "trapezoid")
  double adcMin_, adcMax_, stripNumMin_, stripNumMax_, wgMin_, wgMax_, widthMin_, widthMax_;
  std::array<double, 3> lcMin_, lcMax_;
  std::map<int, Trap> trap_;  // station*10 + raw ring
  // output: simhits_time (z-score), simhits_energy (z-score of log10(E + logEps))
  double timeMean_, timeStd_, energyMean_, energyStd_, energyLogEps_;

  Ort::Env env_;
  std::unique_ptr<Ort::Session> encSession_;
  std::unique_ptr<Ort::Session> decSession_;

  unsigned long long nEvents_ = 0, nDocs_ = 0, nHitsMade_ = 0, sumSlots_ = 0;
  unsigned long long nME11WiresOffered_ = 0, nME11WiresTaken_ = 0;
};

// -----------------------------------------------------------------------------

MLCSCRecHitProducer::MLCSCRecHitProducer(const edm::ParameterSet& iConfig)
    : wireDigiToken_(consumes<CSCWireDigiCollection>(iConfig.getParameter<edm::InputTag>("wireDigiLabel"))),
      stripDigiToken_(consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("stripDigiLabel"))),
      geometryToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()),
      encoderPath_(iConfig.getParameter<std::string>("encoderPath")),
      decoderPath_(iConfig.getParameter<std::string>("decoderPath")),
      nKeepRule_(iConfig.getParameter<std::string>("nKeepRule")),
      slotZ_(iConfig.getParameter<double>("slotZ")),
      stripAdcRangeThreshold_(iConfig.getParameter<double>("stripAdcRangeThreshold")),
      minKeep_(iConfig.getParameter<int>("minKeep")),
      minKeepMinMu_(iConfig.getParameter<double>("minKeepMinMu")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      env_(ORT_LOGGING_LEVEL_WARNING, "MLCSCRecHitProducer") {
  if (encoderPath_.empty() || decoderPath_.empty())
    throw cms::Exception("Configuration") << "encoderPath and decoderPath are required";
  if (nKeepRule_ != "mean" && nKeepRule_ != "upper")
    throw cms::Exception("Configuration") << "nKeepRule must be 'mean' or 'upper', got '" << nKeepRule_ << "'";

  // filled by mlCSCRecHits_cfi.read_norm_stats from the export sidecar <stem>.json
  const auto& n = iConfig.getParameter<edm::ParameterSet>("normalization");
  adcMin_ = n.getParameter<double>("adcMin");
  adcMax_ = n.getParameter<double>("adcMax");
  stripNumMin_ = n.getParameter<double>("stripNumberMin");
  stripNumMax_ = n.getParameter<double>("stripNumberMax");
  wgMin_ = n.getParameter<double>("wireGroupMin");
  wgMax_ = n.getParameter<double>("wireGroupMax");
  widthMin_ = n.getParameter<double>("wireWidthMin");
  widthMax_ = n.getParameter<double>("wireWidthMax");
  const auto lcMin = n.getParameter<std::vector<double>>("layerCentreMin");
  const auto lcMax = n.getParameter<std::vector<double>>("layerCentreMax");
  const auto tKey = n.getParameter<std::vector<int>>("trapKey");
  const auto tVlo = n.getParameter<std::vector<double>>("trapVLo");
  const auto tVhi = n.getParameter<std::vector<double>>("trapVHi");
  const auto tHlo = n.getParameter<std::vector<double>>("trapHwLo");
  const auto tHhi = n.getParameter<std::vector<double>>("trapHwHi");
  if (lcMin.size() != 3 || lcMax.size() != 3 || tKey.empty() || tVlo.size() != tKey.size() ||
      tVhi.size() != tKey.size() || tHlo.size() != tKey.size() || tHhi.size() != tKey.size())
    throw cms::Exception("Configuration") << "normalization PSet incomplete: run read_norm_stats(<stem>.json)";
  std::copy(lcMin.begin(), lcMin.end(), lcMin_.begin());
  std::copy(lcMax.begin(), lcMax.end(), lcMax_.begin());
  for (size_t i = 0; i < tKey.size(); ++i)
    trap_[tKey[i]] = Trap{tVlo[i], tVhi[i], tHlo[i], tHhi[i]};
  timeMean_ = n.getParameter<double>("timeMean");
  timeStd_ = n.getParameter<double>("timeStd");
  energyMean_ = n.getParameter<double>("energyMean");
  energyStd_ = n.getParameter<double>("energyStd");
  energyLogEps_ = n.getParameter<double>("energyLogEps");

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);  // CMSSW owns the threading model
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // shapes change every event: the CPU arena and the per-shape memory plan only grow (held at 18-53 GB 11-02)
  opts.DisableCpuMemArena();
  opts.DisableMemPattern();
  encSession_ = std::make_unique<Ort::Session>(env_, edm::FileInPath(encoderPath_).fullPath().c_str(), opts);
  decSession_ = std::make_unique<Ort::Session>(env_, edm::FileInPath(decoderPath_).fullPath().c_str(), opts);

  // graph contract (export_ragged_cmssw.graph_io): strips_emb A->B and hit_sigma are optional arms
  Ort::AllocatorWithDefaultOptions alloc;
  auto names = [&](Ort::Session& s, bool inputs) {
    std::vector<std::string> v;
    const size_t k = inputs ? s.GetInputCount() : s.GetOutputCount();
    for (size_t i = 0; i < k; ++i)
      v.emplace_back(inputs ? s.GetInputNameAllocated(i, alloc).get() : s.GetOutputNameAllocated(i, alloc).get());
    return v;
  };
  const auto encOut = names(*encSession_, false);
  const auto encIn = names(*encSession_, true);
  const auto decIn = names(*decSession_, true);
  const auto decOut = names(*decSession_, false);
  const bool encStrips = encOut.size() == 5 && encOut[4] == "strips_emb";
  stripsToDecoder_ = decIn.size() == 8 && decIn[6] == "strips_emb" && decIn[7] == "det_idx_strips";
  decoderHasSigma_ = decOut.size() == 3 && decOut[2] == "hit_sigma";
  if (encStrips != stripsToDecoder_ || (encOut.size() != 4 && !encStrips) || (decIn.size() != 6 && !stripsToDecoder_) ||
      (decOut.size() != 2 && !decoderHasSigma_))
    throw cms::Exception("Configuration") << "encoder/decoder graphs do not match a known contract; "
                                          << "re-export both from one run";
  // classical-mix graphs take 25 wire / 16 strip columns; the old v22/v25 graphs 28 / 20
  const auto wShape = encSession_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto sShape = encSession_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
  if (encIn.size() != 7 || wShape.size() != 2 || sShape.size() != 2 || wShape[1] != kNWireFeat ||
      sShape[1] != kNStripFeat)
    throw cms::Exception("Configuration") << "encoder expects " << (wShape.size() == 2 ? wShape[1] : -1) << "/"
                                          << (sShape.size() == 2 ? sShape[1] : -1)
                                          << " wire/strip columns; this producer builds the classical-mix "
                                          << kNWireFeat << "/" << kNStripFeat;

  produces<CSCRecHit2DCollection>();
}

const Trap& MLCSCRecHitProducer::trapOf(uint32_t raw) const {
  const auto it = trap_.find(rawStation(raw) * 10 + rawRing(raw));
  if (it == trap_.end())
    throw cms::Exception("MLCSCRecHitProducer") << "no trapezoid for station " << rawStation(raw) << " raw ring "
                                                << rawRing(raw);
  return it->second;
}

void MLCSCRecHitProducer::endStream() {
  if (nEvents_ == 0)
    return;
  edm::LogInfo("MLCSCRecHitProducer") << "events=" << nEvents_ << " docs=" << nDocs_ << " hits=" << nHitsMade_
                                      << " slots/event=" << std::fixed << std::setprecision(1)
                                      << double(sumSlots_) / nEvents_ << " ME1/1 wires taken/offered="
                                      << nME11WiresTaken_ << "/" << nME11WiresOffered_ << " nKeepRule=" << nKeepRule_
                                      << " minKeep=" << minKeep_ << " stripsToDecoder=" << stripsToDecoder_;
}

void MLCSCRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("wireDigiLabel", edm::InputTag("muonCSCDigis", "MuonCSCWireDigi"));
  desc.add<edm::InputTag>("stripDigiLabel", edm::InputTag("muonCSCDigis", "MuonCSCStripDigi"));
  desc.add<std::string>("encoderPath", "")->setComment("FileInPath to <stem>_encoder.onnx");
  desc.add<std::string>("decoderPath", "")->setComment("FileInPath to <stem>_decoder.onnx");
  desc.add<std::string>("nKeepRule", "mean")
      ->setComment("'mean' = round(count_mu), the offline decode; 'upper' = round(mu + slotZ*sd), the budget");
  desc.add<double>("slotZ", 2.0)->setComment("the run's slot_z; only used by nKeepRule='upper'");
  desc.add<double>("stripAdcRangeThreshold", 13.3)->setComment("fired strip: max(ADC)-min(ADC) > this (training)");
  desc.add<int>("minKeep", 0)->setComment("per-doc floor on kept hits; 0 = plain nKeepRule");
  desc.add<double>("minKeepMinMu", 0.0)->setComment("apply minKeep only to docs with count mu >= this");

  edm::ParameterSetDescription norm;
  for (const char* k : {"adcMin", "adcMax", "stripNumberMin", "stripNumberMax", "wireGroupMin", "wireGroupMax",
                        "wireWidthMin", "wireWidthMax", "timeMean", "timeStd", "energyMean", "energyStd"})
    norm.add<double>(k, 0.);
  norm.add<double>("energyLogEps", 1e-12);
  norm.add<std::vector<double>>("layerCentreMin", {});
  norm.add<std::vector<double>>("layerCentreMax", {});
  norm.add<std::vector<int>>("trapKey", {});
  for (const char* k : {"trapVLo", "trapVHi", "trapHwLo", "trapHwHi"})
    norm.add<std::vector<double>>(k, {});
  desc.add<edm::ParameterSetDescription>("normalization", norm)
      ->setComment("classical-mix stats, from the export sidecar via mlCSCRecHits_cfi.read_norm_stats");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("mlCSCRecHits", desc);
}

// -----------------------------------------------------------------------------

void MLCSCRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geo = iSetup.getData(geometryToken_);
  const auto& wireDigis = iEvent.get(wireDigiToken_);
  const auto& stripDigis = iEvent.get(stripDigiToken_);
  nEvents_++;
  auto out = std::make_unique<CSCRecHit2DCollection>();

  std::map<uint32_t, Frame> frames;
  auto frame = [&](uint32_t raw) -> const Frame* {
    auto it = frames.find(raw);
    if (it != frames.end())
      return &it->second;
    const CSCLayer* l = geo.layer(CSCDetId(raw));
    return l ? &frames.emplace(raw, frameOf(l)).first->second : nullptr;
  };

  // --- strips: fired = ADC range > threshold over all samples (ntupler) and over the first 8 (packer)
  std::vector<StripRow> strips;
  for (auto detIt = stripDigis.begin(); detIt != stripDigis.end(); ++detIt) {
    const CSCDetId& id = (*detIt).first;
    const CSCLayer* layer = geo.layer(id);
    const Frame* f = layer ? frame(id.rawId()) : nullptr;
    if (!f)
      continue;
    const CSCLayerGeometry* lg = layer->geometry();
    const auto ylim = lg->yLimitsOfStripPlane();
    for (auto d = (*detIt).second.first; d != (*detIt).second.second; ++d) {
      const std::vector<int>& adc = d->getADCCounts();
      if (adc.empty())
        continue;
      const auto mmAll = std::minmax_element(adc.begin(), adc.end());
      if (*mmAll.second - *mmAll.first <= stripAdcRangeThreshold_)
        continue;
      StripRow r;
      r.adc.fill(0.);
      for (unsigned i = 0; i < std::min<size_t>(kNAdc, adc.size()); ++i)
        r.adc[i] = adc[i];
      const auto mm8 = std::minmax_element(r.adc.begin(), r.adc.end());
      if (*mm8.second - *mm8.first <= stripAdcRangeThreshold_)
        continue;
      r.raw = id.rawId();
      r.strip = d->getStrip();
      // radial strip line: x of the strip at both y limits of the strip plane (ntupler, fixed 10-26)
      const GlobalPoint g0 = layer->toGlobal(LocalPoint(lg->xOfStrip(r.strip, ylim.first), ylim.first, 0.));
      const GlobalPoint g1 = layer->toGlobal(LocalPoint(lg->xOfStrip(r.strip, ylim.second), ylim.second, 0.));
      f->toLocal(float(g0.x()), float(g0.y()), r.u0, r.v0);
      f->toLocal(float(g1.x()), float(g1.y()), r.u1, r.v1);
      strips.push_back(r);
    }
  }

  // --- wires: middle wire of the group from the wire topology (tilted in ME1/1), width = nWires * pitch
  std::vector<WireRow> wires;
  for (auto detIt = wireDigis.begin(); detIt != wireDigis.end(); ++detIt) {
    const CSCDetId& id = (*detIt).first;
    const CSCLayer* layer = geo.layer(id);
    const Frame* f = layer ? frame(id.rawId()) : nullptr;
    if (!f)
      continue;
    const CSCLayerGeometry* lg = layer->geometry();
    for (auto d = (*detIt).second.first; d != (*detIt).second.second; ++d) {
      WireRow r;
      r.raw = id.rawId();
      r.group = d->getWireGroup();
      r.tb.fill(0.);
      const std::vector<int> on = d->getTimeBinsOn();  // 0-based: column i = bin i
      for (unsigned q = 0; q < std::min<size_t>(kMaxTimeBinsOn, on.size()); ++q)
        if (on[q] >= 0 && on[q] < kNTimeBins)
          r.tb[on[q]] = 1.;
      const auto ends = lg->wireTopology()->wireEnds(lg->middleWireOfGroup(r.group));
      const GlobalPoint g0 = layer->toGlobal(ends.first), g1 = layer->toGlobal(ends.second);
      r.gx0 = float(g0.x());
      r.gy0 = float(g0.y());
      r.gx1 = float(g1.x());
      r.gy1 = float(g1.y());
      f->toLocal(r.gx0, r.gy0, r.u0, r.v0);
      f->toLocal(r.gx1, r.gy1, r.u1, r.v1);
      r.width = float(lg->numberOfWiresPerGroup(r.group) * lg->wirePitch());
      wires.push_back(r);
    }
  }

  // --- docs (csc_event_datamaker.process_event): a layer with fired strips and wires; ME1/1 by wire band
  std::map<uint32_t, std::vector<size_t>> stripsOf, wiresOf;
  for (size_t i = 0; i < strips.size(); ++i)
    stripsOf[strips[i].raw].push_back(i);
  std::map<uint32_t, std::vector<size_t>> wiresByLayer;
  for (size_t i = 0; i < wires.size(); ++i)
    wiresByLayer[wires[i].raw].push_back(i);
  std::vector<uint32_t> docs;  // ascending rawId (std::map order)
  for (const auto& [raw, sIdx] : stripsOf) {
    const uint32_t wk = isME11(raw) ? toME11b(raw) : raw;
    const auto wit = wiresByLayer.find(wk);
    if (wit == wiresByLayer.end())
      continue;
    if (!isME11(raw)) {
      wiresOf[raw] = wit->second;
      docs.push_back(raw);
      continue;
    }
    double vlo = 1e30, vhi = -1e30;
    for (size_t i : sIdx) {
      vlo = std::min({vlo, strips[i].v0, strips[i].v1});
      vhi = std::max({vhi, strips[i].v0, strips[i].v1});
    }
    std::vector<size_t> sel;
    for (size_t i : wit->second) {
      const auto band = wireBandV(wires[i]);
      if (band.second >= vlo && band.first <= vhi)
        sel.push_back(i);
    }
    nME11WiresOffered_ += wit->second.size();
    nME11WiresTaken_ += sel.size();
    if (!sel.empty()) {
      wiresOf[raw] = sel;
      docs.push_back(raw);
    }
  }
  const size_t nDocs = docs.size();
  if (nDocs == 0) {
    iEvent.put(std::move(out));
    return;
  }

  // --- tensors (classicalmix_dataset.__getitem__), doc-major, strips by strip number, wires by group
  std::vector<float> wireFeat, stripFeat;
  std::vector<int64_t> docIdxW, docIdxS, detW, detS, detDocs(docs.begin(), docs.end());
  for (size_t d = 0; d < nDocs; ++d) {
    const uint32_t raw = docs[d];
    const Frame& f = *frame(raw);
    const Trap& T = trapOf(raw);
    const std::array<double, 3> c{{f.cx, f.cy, f.cz}};
    std::array<double, 3> cn;
    for (int k = 0; k < 3; ++k)
      cn[k] = minmax(c[k], lcMin_[k], lcMax_[k]);

    auto sIdx = stripsOf[raw];
    std::stable_sort(sIdx.begin(), sIdx.end(), [&](size_t a, size_t b) { return strips[a].strip < strips[b].strip; });
    for (size_t i : sIdx) {
      const StripRow& s = strips[i];
      double un0, vn0, un1, vn1;
      T.norm(s.u0, s.v0, un0, vn0);
      T.norm(s.u1, s.v1, un1, vn1);
      for (unsigned k = 0; k < kNAdc; ++k)
        stripFeat.push_back(minmax(s.adc[k], adcMin_, adcMax_));
      for (double v : {minmax(s.strip, stripNumMin_, stripNumMax_), un0, vn0, un1, vn1, cn[0], cn[1], cn[2]})
        stripFeat.push_back(v);
      docIdxS.push_back(d);
      detS.push_back(raw);
    }

    auto wIdx = wiresOf[raw];
    std::stable_sort(wIdx.begin(), wIdx.end(), [&](size_t a, size_t b) { return wires[a].group < wires[b].group; });
    for (size_t i : wIdx) {
      const WireRow& w = wires[i];
      double u0 = w.u0, v0 = w.v0, u1 = w.u1, v1 = w.v1;
      if (w.raw != raw) {  // ME1/1b wire copied into an ME1/a doc: coordinates in the doc's frame
        f.toLocal(w.gx0, w.gy0, u0, v0);
        f.toLocal(w.gx1, w.gy1, u1, v1);
      }
      double un0, vn0, un1, vn1;
      T.norm(u0, v0, un0, vn0);
      T.norm(u1, v1, un1, vn1);
      for (int k = 0; k < kNTimeBins; ++k)
        wireFeat.push_back(w.tb[k]);
      for (double v : {minmax(w.group, wgMin_, wgMax_), un0, vn0, un1, vn1, minmax(w.width, widthMin_, widthMax_), cn[0],
                       cn[1], cn[2]})
        wireFeat.push_back(v);
      docIdxW.push_back(d);
      detW.push_back(raw);
    }
  }
  nDocs_ += nDocs;

  // --- graph A: encoder + count head
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t nW = docIdxW.size(), nS = docIdxS.size(), nD = nDocs;
  const std::array<int64_t, 2> shW{{nW, kNWireFeat}}, shS{{nS, kNStripFeat}};
  const std::array<int64_t, 1> shIW{{nW}}, shIS{{nS}}, shD{{nD}};
  static const char* kEncIn[] = {"wires", "strips", "doc_idx_wires", "doc_idx_strips",
                                 "det_idx_wires", "det_idx_strips", "det_idx_docs"};
  static const char* kEncOut[] = {"wires_out", "doc_cond", "count_mu", "n_slots", "strips_emb"};
  std::vector<Ort::Value> encIn;
  encIn.push_back(Ort::Value::CreateTensor<float>(mem, wireFeat.data(), wireFeat.size(), shW.data(), 2));
  encIn.push_back(Ort::Value::CreateTensor<float>(mem, stripFeat.data(), stripFeat.size(), shS.data(), 2));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docIdxW.data(), docIdxW.size(), shIW.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docIdxS.data(), docIdxS.size(), shIS.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, detW.data(), detW.size(), shIW.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, detS.data(), detS.size(), shIS.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, detDocs.data(), detDocs.size(), shD.data(), 1));
  auto encOut = encSession_->Run(
      Ort::RunOptions{nullptr}, kEncIn, encIn.data(), encIn.size(), kEncOut, stripsToDecoder_ ? 5 : 4);
  const float* countMu = encOut[2].GetTensorData<float>();
  const int32_t* nSlots = encOut[3].GetTensorData<int32_t>();

  // --- packed slots (export_ragged_cmssw.pack_slots): `within` restarts at 0 per doc
  std::vector<int64_t> docOfSlot, within, hitDet;
  std::vector<size_t> docStart(nDocs, 0);
  for (size_t d = 0; d < nDocs; ++d) {
    docStart[d] = docOfSlot.size();
    for (int j = 0; j < std::max(0, nSlots[d]); ++j) {
      docOfSlot.push_back(d);
      within.push_back(j);
      hitDet.push_back(detDocs[d]);
    }
  }
  const size_t totalSlots = docOfSlot.size();
  if (totalSlots == 0) {
    iEvent.put(std::move(out));
    return;
  }
  sumSlots_ += totalSlots;

  // --- graph B: decoder
  const std::array<int64_t, 1> shSlot{{int64_t(totalSlots)}};
  static const char* kDecIn[] = {"wires_out", "doc_cond", "det_idx_wires", "doc_of_slot",
                                 "within", "hit_det_idx", "strips_emb", "det_idx_strips"};
  static const char* kDecOut[] = {"hits", "exist_logit", "hit_sigma"};
  std::vector<Ort::Value> decIn;
  decIn.push_back(std::move(encOut[0]));
  decIn.push_back(std::move(encOut[1]));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, detW.data(), detW.size(), shIW.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docOfSlot.data(), totalSlots, shSlot.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, within.data(), totalSlots, shSlot.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, hitDet.data(), totalSlots, shSlot.data(), 1));
  if (stripsToDecoder_) {
    decIn.push_back(std::move(encOut[4]));
    decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, detS.data(), detS.size(), shIS.data(), 1));
  }
  auto decOut = decSession_->Run(
      Ort::RunOptions{nullptr}, kDecIn, decIn.data(), decIn.size(), kDecOut, decoderHasSigma_ ? 3 : 2);
  const float* hits = decOut[0].GetTensorData<float>();        // (S, 4) u_n, v_n, t_z, E_z
  const float* existLogit = decOut[1].GetTensorData<float>();  // (S,)

  // --- per doc: rank by exist logit, keep K; invert as denormalize_target
  std::vector<std::pair<CSCDetId, CSCRecHit2D>> made;
  for (size_t d = 0; d < nDocs; ++d) {
    const int ns = std::max(0, nSlots[d]);
    const double mu = countMu[d];
    double k = mu;
    if (nKeepRule_ == "upper")
      k = mu + slotZ_ * std::max((ns - mu) / std::max(slotZ_, 1e-6), 0.);
    int keep = int(std::lround(k));
    if (minKeep_ > 0 && mu >= minKeepMinMu_)
      keep = std::max(keep, minKeep_);
    keep = std::max(0, std::min(keep, ns));

    std::vector<size_t> order(ns);
    for (int j = 0; j < ns; ++j)
      order[j] = docStart[d] + j;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return existLogit[a] > existLogit[b]; });

    const uint32_t raw = docs[d];
    const CSCDetId id(raw);
    const CSCLayer* layer = geo.layer(id);
    const Frame& f = *frame(raw);
    const Trap& T = trapOf(raw);
    for (int r = 0; r < keep; ++r) {
      const float* h = hits + 4 * order[r];
      double u, v, gx, gy;
      T.denorm(h[0], h[1], u, v);
      f.toGlobal(u, v, gx, gy);
      const double t = h[2] * timeStd_ + timeMean_;  // ns, TOF - OFF (prompt muon = 0)
      const double e = std::pow(10.0, h[3] * energyStd_ + energyMean_) - energyLogEps_;
      const LocalPoint lp = layer->toLocal(GlobalPoint(gx, gy, f.cz));  // z = layer plane (model z-LUT)

      // placeholders: no strip / wire identity. channels must stay one element (CSCRecHit2D divides by its size);
      // hitWire() == 0 tags an ML hit (real wires are 1-based)
      CSCRecHit2D::ChannelContainer channels{0};
      CSCRecHit2D::ADCContainer adcs;
      CSCRecHit2D::ChannelContainer wgroups;
      made.emplace_back(id,
                        CSCRecHit2D(id,
                                    lp,
                                    LocalError(1.f, 0.f, 1.f),  // dummy until a sigma table exists
                                    channels,
                                    adcs,
                                    wgroups,
                                    float(t),  // tpeak
                                    0.f,
                                    0.f,
                                    0,
                                    0,
                                    0,
                                    int(std::lround(t * 100.0)),  // wireTime, stored x100
                                    float(e)));
      nHitsMade_++;
    }
  }

  // RangeMap: insert grouped by DetId
  std::stable_sort(made.begin(), made.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (auto it = made.begin(); it != made.end();) {
    auto jt = it;
    std::vector<CSCRecHit2D> group;
    for (; jt != made.end() && jt->first == it->first; ++jt)
      group.push_back(jt->second);
    out->put(it->first, group.begin(), group.end());
    it = jt;
  }
  if (verbose_)
    edm::LogInfo("MLCSCRecHitProducer") << "docs=" << nDocs << " wires=" << nW << " strips=" << nS
                                        << " slots=" << totalSlots << " hits=" << made.size();
  iEvent.put(std::move(out));
}

DEFINE_FWK_MODULE(MLCSCRecHitProducer);
