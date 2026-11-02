// -*- C++ -*-
//
// Package:    HMTntuple/CSCShowerAnalyzer
// Class:      MLDTRecHitProducer
//
/**\class MLDTRecHitProducer MLDTRecHitProducer.cc

  Runs the DT hit-regression network (flex_atten/dt, dt_cm_v*) on DT digis and emits the
  predictions as a DTRecHitCollection, the DT twin of MLCSCRecHitProducer.

  Inputs reproduce dt/dt_event_datamaker.py + dt/dt_dataset.py: one token per cell (layer, wire)
  with up to 4 TDC hits in time order; drift time = digi time - DTTTrigSync offset at the wire
  centre (as dt1DRecHits / the ntupler). A doc is a layer with >= 1 digi. Two ONNX graphs from
  mds-ml/onnx/export_dt_cmssw.py; outputs inverted as dt/test_dt.denorm().

  Output: each ML hit is a DTRecHit1DPair whose left AND right hits sit at the predicted layer-local
  (x, y, 0), so localPosition() is the full 2-D point (y = along the wire, which real 1-D rechits do
  not have; MB4 is trained to the wire centre, y ~ 0). wireId = nearest wire to x, digiTime = ML time
  in ns (TOF - OFF_DT: prompt muon = 0), stored in TDC counts (25/32 ns).
  DTRecHit1DPair has no energy field, so per-hit energy (GeV) and the unquantised time (ns) go out as
  std::vector<float> "energy" / "time", in the collection's iteration order (index = Ref key).
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace {
  constexpr int kMaxHits = 4;               // dt_event_datamaker.MAX_HITS
  constexpr int kNFeat = 2 * kMaxHits + 4;  // drift_0..3, filled_0..3, u_n, cx, cy, cz
  constexpr int kNCat = 3;                  // station-1, superlayer-1, layer-1

  struct Cell {
    uint32_t layer;
    int wire;
    std::vector<std::pair<double, double>> hits;  // (TDC time, drift time) ns
  };
}  // namespace

class MLDTRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit MLDTRecHitProducer(const edm::ParameterSet&);
  ~MLDTRecHitProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  const edm::EDGetTokenT<DTDigiCollection> digiToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> geometryToken_;
  std::unique_ptr<DTTTrigBaseSync> sync_;
  const std::string encoderPath_, decoderPath_, nKeepRule_;
  const double slotZ_;
  const int minKeep_;
  const double minKeepMinMu_;
  const bool verbose_;
  // dt_dataset.DT_NORM (settings["dt_norm"]), from the export sidecar
  double driftTScale_, timeScale_, logEMean_, logEStd_, rScale_, zScale_;

  Ort::Env env_;
  std::unique_ptr<Ort::Session> encSession_, decSession_;
  unsigned long long nEvents_ = 0, nDocs_ = 0, nCells_ = 0, nHitsOverMax_ = 0, nHitsMade_ = 0, sumSlots_ = 0;
};

MLDTRecHitProducer::MLDTRecHitProducer(const edm::ParameterSet& iConfig)
    : digiToken_(consumes<DTDigiCollection>(iConfig.getParameter<edm::InputTag>("digiLabel"))),
      geometryToken_(esConsumes<DTGeometry, MuonGeometryRecord>()),
      sync_(DTTTrigSyncFactory::get()->create(iConfig.getParameter<std::string>("tTrigMode"),
                                              iConfig.getParameter<edm::ParameterSet>("tTrigModeConfig"),
                                              consumesCollector())),
      encoderPath_(iConfig.getParameter<std::string>("encoderPath")),
      decoderPath_(iConfig.getParameter<std::string>("decoderPath")),
      nKeepRule_(iConfig.getParameter<std::string>("nKeepRule")),
      slotZ_(iConfig.getParameter<double>("slotZ")),
      minKeep_(iConfig.getParameter<int>("minKeep")),
      minKeepMinMu_(iConfig.getParameter<double>("minKeepMinMu")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      env_(ORT_LOGGING_LEVEL_WARNING, "MLDTRecHitProducer") {
  if (nKeepRule_ != "mean" && nKeepRule_ != "upper")
    throw cms::Exception("Configuration") << "nKeepRule must be 'mean' or 'upper'";
  const auto& n = iConfig.getParameter<edm::ParameterSet>("normalization");
  driftTScale_ = n.getParameter<double>("drift_tscale");
  timeScale_ = n.getParameter<double>("time_scale");
  logEMean_ = n.getParameter<double>("loge_mean");
  logEStd_ = n.getParameter<double>("loge_std");
  rScale_ = n.getParameter<double>("r_scale");
  zScale_ = n.getParameter<double>("z_scale");
  if (driftTScale_ <= 0 || timeScale_ <= 0 || rScale_ <= 0 || zScale_ <= 0)
    throw cms::Exception("Configuration") << "normalization PSet empty: run mlDTRecHits_cfi.read_dt_norm(<stem>.json)";

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // shapes change every event: the CPU arena and the per-shape memory plan only grow (held at 18-53 GB 11-02)
  opts.DisableCpuMemArena();
  opts.DisableMemPattern();
  encSession_ = std::make_unique<Ort::Session>(env_, edm::FileInPath(encoderPath_).fullPath().c_str(), opts);
  decSession_ = std::make_unique<Ort::Session>(env_, edm::FileInPath(decoderPath_).fullPath().c_str(), opts);
  const auto shp = encSession_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (encSession_->GetInputCount() != 5 || decSession_->GetInputCount() != 6 || shp.size() != 2 || shp[1] != kNFeat)
    throw cms::Exception("Configuration") << "DT graphs do not match export_dt_cmssw.py's contract";
  produces<DTRecHitCollection>();
  produces<std::vector<float>>("energy");
  produces<std::vector<float>>("time");
}

void MLDTRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiLabel", edm::InputTag("muonDTDigis"))->setComment("unpacked DT digis (training)");
  desc.add<std::string>("tTrigMode", "DTTTrigSyncFromDB");
  edm::ParameterSetDescription syncDesc;
  syncDesc.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("tTrigModeConfig", syncDesc)
      ->setComment("copied from dt1DRecHits.recAlgoConfig, as the ntupler");
  desc.add<std::string>("encoderPath", "");
  desc.add<std::string>("decoderPath", "");
  desc.add<std::string>("nKeepRule", "mean");
  desc.add<double>("slotZ", 2.0);
  desc.add<int>("minKeep", 0)->setComment("per-doc floor on kept hits; 0 = plain nKeepRule");
  desc.add<double>("minKeepMinMu", 0.0)->setComment("apply minKeep only to docs with count mu >= this");
  edm::ParameterSetDescription norm;
  for (const char* k : {"drift_tscale", "time_scale", "loge_mean", "loge_std", "r_scale", "z_scale"})
    norm.add<double>(k, 0.);
  desc.add<edm::ParameterSetDescription>("normalization", norm);
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("mlDTRecHits", desc);
}

void MLDTRecHitProducer::endStream() {
  if (nEvents_ == 0)
    return;
  edm::LogInfo("MLDTRecHitProducer") << "events=" << nEvents_ << " docs=" << nDocs_ << " cells=" << nCells_
                                     << " TDC hits beyond " << kMaxHits << "/cell=" << nHitsOverMax_
                                     << " hits=" << nHitsMade_ << " slots/event=" << std::fixed
                                     << std::setprecision(1) << double(sumSlots_) / nEvents_
                                     << " nKeepRule=" << nKeepRule_ << " minKeep=" << minKeep_;
}

void MLDTRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geo = iSetup.getData(geometryToken_);
  const auto& digis = iEvent.get(digiToken_);
  sync_->setES(iSetup);
  nEvents_++;
  auto out = std::make_unique<DTRecHitCollection>();
  auto outE = std::make_unique<std::vector<float>>();
  auto outT = std::make_unique<std::vector<float>>();
  auto putAll = [&]() {
    iEvent.put(std::move(out));
    iEvent.put(std::move(outE), "energy");
    iEvent.put(std::move(outT), "time");
  };

  // cells keyed (layer, wire): std::map gives the packer's (layer, wire) order
  std::map<std::pair<uint32_t, int>, Cell> cells;
  for (auto detIt = digis.begin(); detIt != digis.end(); ++detIt) {
    const DTLayerId& lid = (*detIt).first;
    const DTLayer* layer = geo.layer(lid);
    if (!layer)
      continue;
    const DTTopology& topo = layer->specificTopology();
    for (auto d = (*detIt).second.first; d != (*detIt).second.second; ++d) {
      const int wire = d->wire();
      const GlobalPoint c = layer->toGlobal(LocalPoint(topo.wirePosition(wire), 0., 0.));
      Cell& cell = cells[{lid.rawId(), wire}];
      cell.layer = lid.rawId();
      cell.wire = wire;
      cell.hits.emplace_back(d->time(), d->time() - sync_->offset(layer, DTWireId(lid, wire), c));
    }
  }
  if (cells.empty()) {
    putAll();
    return;
  }

  std::vector<uint32_t> docs;
  for (const auto& kv : cells)
    if (docs.empty() || docs.back() != kv.second.layer)
      docs.push_back(kv.second.layer);
  std::map<uint32_t, int64_t> docIndex;
  for (size_t i = 0; i < docs.size(); ++i)
    docIndex[docs[i]] = i;

  std::vector<float> feat;
  std::vector<int64_t> cat, det, docOf, docDet(docs.begin(), docs.end());
  for (auto& kv : cells) {
    Cell& cl = kv.second;
    const DTLayerId lid(cl.layer);
    const DTLayer* layer = geo.layer(lid);
    std::stable_sort(cl.hits.begin(), cl.hits.end());  // time order (packer: lexsort by time)
    if (int(cl.hits.size()) > kMaxHits)
      nHitsOverMax_ += cl.hits.size() - kMaxHits;
    for (int k = 0; k < kMaxHits; ++k)
      feat.push_back(k < int(cl.hits.size()) ? cl.hits[k].second / driftTScale_ : 0.);
    for (int k = 0; k < kMaxHits; ++k)
      feat.push_back(k < int(cl.hits.size()) ? 1. : 0.);
    // u = wire centre in the layer frame (ntupler stores the global centre as float32)
    const GlobalPoint c = layer->toGlobal(LocalPoint(layer->specificTopology().wirePosition(cl.wire), 0., 0.));
    const double u = layer->toLocal(GlobalPoint(float(c.x()), float(c.y()), float(c.z()))).x();
    const GlobalPoint o = layer->toGlobal(LocalPoint(0., 0., 0.));
    feat.push_back(u / (0.5 * layer->surface().bounds().width()));
    feat.push_back(o.x() / rScale_);
    feat.push_back(o.y() / rScale_);
    feat.push_back(o.z() / zScale_);
    cat.push_back(lid.station() - 1);
    cat.push_back(lid.superlayer() - 1);
    cat.push_back(lid.layer() - 1);
    det.push_back(cl.layer);
    docOf.push_back(docIndex[cl.layer]);
  }
  const size_t nDocs = docs.size(), nCell = det.size();
  nDocs_ += nDocs;
  nCells_ += nCell;

  // --- graph A
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 2> shF{{int64_t(nCell), kNFeat}}, shC{{int64_t(nCell), kNCat}};
  const std::array<int64_t, 1> shN{{int64_t(nCell)}}, shD{{int64_t(nDocs)}};
  static const char* kEncIn[] = {"digi_feat", "digi_cat", "digi_det", "digi_doc", "doc_det"};
  static const char* kEncOut[] = {"digi_out", "doc_cond", "count_mu", "n_slots"};
  std::vector<Ort::Value> encIn;
  encIn.push_back(Ort::Value::CreateTensor<float>(mem, feat.data(), feat.size(), shF.data(), 2));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, cat.data(), cat.size(), shC.data(), 2));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, det.data(), nCell, shN.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docOf.data(), nCell, shN.data(), 1));
  encIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docDet.data(), nDocs, shD.data(), 1));
  auto encOut = encSession_->Run(Ort::RunOptions{nullptr}, kEncIn, encIn.data(), encIn.size(), kEncOut, 4);
  const float* countMu = encOut[2].GetTensorData<float>();
  const int32_t* nSlots = encOut[3].GetTensorData<int32_t>();

  std::vector<int64_t> docOfSlot, within, hitDet;
  std::vector<size_t> docStart(nDocs);
  for (size_t d = 0; d < nDocs; ++d) {
    docStart[d] = docOfSlot.size();
    for (int j = 0; j < std::max(0, nSlots[d]); ++j) {
      docOfSlot.push_back(d);
      within.push_back(j);
      hitDet.push_back(docDet[d]);
    }
  }
  const size_t S = docOfSlot.size();
  if (S == 0) {
    putAll();
    return;
  }
  sumSlots_ += S;

  // --- graph B
  const std::array<int64_t, 1> shS{{int64_t(S)}};
  static const char* kDecIn[] = {"digi_out", "doc_cond", "digi_det", "doc_of_slot", "within", "hit_det"};
  static const char* kDecOut[] = {"hits", "exist_logit"};
  std::vector<Ort::Value> decIn;
  decIn.push_back(std::move(encOut[0]));
  decIn.push_back(std::move(encOut[1]));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, det.data(), nCell, shN.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, docOfSlot.data(), S, shS.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, within.data(), S, shS.data(), 1));
  decIn.push_back(Ort::Value::CreateTensor<int64_t>(mem, hitDet.data(), S, shS.data(), 1));
  auto decOut = decSession_->Run(Ort::RunOptions{nullptr}, kDecIn, decIn.data(), decIn.size(), kDecOut, 2);
  const float* hits = decOut[0].GetTensorData<float>();  // (S, 4) x_n, y_n, t_n, e_n
  const float* logit = decOut[1].GetTensorData<float>();

  for (size_t d = 0; d < nDocs; ++d) {
    const int ns = std::max(0, nSlots[d]);
    const double mu = countMu[d];
    const double k = nKeepRule_ == "upper" ? double(ns) : mu;  // ns = ceil(mu + slotZ*sd)
    int keep = int(std::lround(k));
    if (minKeep_ > 0 && mu >= minKeepMinMu_)
      keep = std::max(keep, minKeep_);
    keep = std::max(0, std::min(keep, ns));
    std::vector<size_t> order(ns);
    for (int j = 0; j < ns; ++j)
      order[j] = docStart[d] + j;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logit[a] > logit[b]; });

    const DTLayerId lid(docs[d]);
    const DTLayer* layer = geo.layer(lid);
    const DTTopology& topo = layer->specificTopology();
    const double hw = 0.5 * layer->surface().bounds().width(), hl = 0.5 * layer->surface().bounds().length();
    edm::OwnVector<DTRecHit1DPair> made;
    for (int r = 0; r < keep; ++r) {
      const float* h = hits + 4 * order[r];
      const LocalPoint lp(h[0] * hw, h[1] * hl, 0.);
      const double t = h[2] * timeScale_;  // ns
      const int wire = std::clamp(topo.channel(lp), topo.firstChannel(), topo.lastChannel());
      DTRecHit1DPair pair(DTWireId(lid, wire), DTDigi(wire, t, 0));
      const LocalError err(1.f, 0.f, 1.f);  // dummy until a sigma table exists
      pair.setPositionAndError(DTEnums::Left, lp, err);
      pair.setPositionAndError(DTEnums::Right, lp, err);
      made.push_back(pair);
      // docs ascend in rawId, so put() order = iteration order = these vectors' order
      outE->push_back(std::pow(10., h[3] * logEStd_ + logEMean_));
      outT->push_back(t);
      nHitsMade_++;
    }
    if (!made.empty())
      out->put(lid, made.begin(), made.end());
  }
  if (verbose_)
    edm::LogInfo("MLDTRecHitProducer") << "docs=" << nDocs << " cells=" << nCell << " slots=" << S;
  putAll();
}

DEFINE_FWK_MODULE(MLDTRecHitProducer);
