// -*- C++ -*-
//
// Joint CSC + DT shower clustering: one fastjet Cambridge/Aachen pass over the CSC and DT hits of an
// event together, so a shower crossing the barrel/endcap overlap or reaching both detectors is ONE
// cluster. Same algorithm and defaults as the stock ca4CSCrechitClusters / ca4DTrechitClusters
// (R = 0.4 in (eta, phi), nRechitMin 50, nStationThres 10); with one detector empty it reduces to them.
//
// Built for the ML hits (mlCSCRecHits + mlDTRecHits): both carry time in ns as TOF - OFF with the prompt
// muon at 0, so CSC and DT times average directly (CSC tpeak, DT digiTime). Real rechits do not share a
// time convention; clustering them here is fine, the cluster time is not.
// Hits go to global with their LAYER frame (the stock HLT clusterer uses the chamber frame).
//
// Emits FlatTables <name> (clusters) and, with saveHits, <name>Hits (constituents, clusterIdx).

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

namespace {
  struct Hit {
    float x, y, z, t, e;
    bool dt;
    int station, ring;  // DT: ring = wheel
  };
}  // namespace

class muonMDSClusterTableProducer : public edm::global::EDProducer<> {
public:
  explicit muonMDSClusterTableProducer(const edm::ParameterSet& c)
      : cscGeoToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()),
        dtGeoToken_(esConsumes<DTGeometry, MuonGeometryRecord>()),
        cscToken_(consumes<CSCRecHit2DCollection>(c.getParameter<edm::InputTag>("cscRecHitLabel"))),
        dtToken_(consumes<DTRecHitCollection>(c.getParameter<edm::InputTag>("dtRecHitLabel"))),
        rParam_(c.getParameter<double>("rParam")),
        nRechitMin_(c.getParameter<int>("nRechitMin")),
        nStationThres_(c.getParameter<int>("nStationThres")),
        saveHits_(c.getParameter<bool>("saveHits")),
        name_(c.getParameter<std::string>("name")) {
    const auto info = c.getParameter<std::string>("dtHitInfoLabel");
    if (!info.empty()) {
      dtEnergyToken_ = consumes<std::vector<float>>(edm::InputTag(info, "energy"));
      dtTimeToken_ = consumes<std::vector<float>>(edm::InputTag(info, "time"));
    }
    produces<nanoaod::FlatTable>(name_);
    if (saveHits_)
      produces<nanoaod::FlatTable>(name_ + "Hits");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("cscRecHitLabel", edm::InputTag("mlCSCRecHits"));
    desc.add<edm::InputTag>("dtRecHitLabel", edm::InputTag("mlDTRecHits"));
    desc.add<std::string>("dtHitInfoLabel", "mlDTRecHits")
        ->setComment("module with per-DT-hit std::vector<float> energy/time; empty = digiTime, no energy");
    desc.add<double>("rParam", 0.4);
    desc.add<int>("nRechitMin", 50);
    desc.add<int>("nStationThres", 10);
    desc.add<bool>("saveHits", false);
    desc.add<std::string>("name", "muonMDSClusterML");
    descriptions.add("muonMDSClusterTable", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeoToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeoToken_;
  const edm::EDGetTokenT<CSCRecHit2DCollection> cscToken_;
  const edm::EDGetTokenT<DTRecHitCollection> dtToken_;
  edm::EDGetTokenT<std::vector<float>> dtEnergyToken_, dtTimeToken_;
  const double rParam_;
  const int nRechitMin_, nStationThres_;
  const bool saveHits_;
  const std::string name_;
};

void muonMDSClusterTableProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  const auto& cscGeo = es.getData(cscGeoToken_);
  const auto& dtGeo = es.getData(dtGeoToken_);
  std::vector<Hit> hits;
  for (const auto& h : ev.get(cscToken_)) {
    const CSCDetId id = h.cscDetId();
    const auto* layer = cscGeo.layer(id);
    if (!layer)
      continue;
    const GlobalPoint g = layer->toGlobal(h.localPosition());
    const int ring = (id.station() == 1 && id.ring() == 4) ? 1 : id.ring();  // ME1/a counts as ME1/1
    hits.push_back({g.x(), g.y(), g.z(), h.tpeak(), h.energyDepositedInLayer(), false, id.station(), ring});
  }
  const auto& dtHits = ev.get(dtToken_);
  const std::vector<float>* dtE = dtEnergyToken_.isUninitialized() ? nullptr : &ev.get(dtEnergyToken_);
  const std::vector<float>* dtT = dtTimeToken_.isUninitialized() ? nullptr : &ev.get(dtTimeToken_);
  if (dtE && (dtE->size() != dtHits.size() || dtT->size() != dtHits.size()))
    throw cms::Exception("LogicError") << "DT hit info size " << dtE->size() << " != DT hits " << dtHits.size();
  size_t k = 0;
  for (const auto& h : dtHits) {
    const size_t key = k++;
    const DTLayerId id(h.geographicalId());
    const auto* layer = dtGeo.layer(id);
    if (!layer)
      continue;
    const GlobalPoint g = layer->toGlobal(h.localPosition());
    hits.push_back({g.x(), g.y(), g.z(), dtT ? (*dtT)[key] : float(h.digiTime()), dtE ? (*dtE)[key] : -1.f, true,
                    id.station(), id.wheel()});
  }

  std::vector<fastjet::PseudoJet> in;
  for (size_t i = 0; i < hits.size(); ++i) {
    const Hit& h = hits[i];
    in.emplace_back(h.x, h.y, h.z, std::sqrt(h.x * h.x + h.y * h.y + h.z * h.z));
    in.back().set_user_index(i);
  }
  fastjet::ClusterSequence seq(in, fastjet::JetDefinition(fastjet::cambridge_algorithm, rParam_));

  std::vector<float> eta, phi, x, y, z, r, avgSt, time, tSpread, tCsc, tDt;
  std::vector<int> size, nCsc, nDt, nSt, nME11, nME12, nME41, nME42, nMB1, nMB2;
  std::vector<int> hCl;
  std::vector<float> hx, hy, hz, ht, he;
  std::vector<int> hDt, hSt, hRing;
  for (const auto& jet : seq.inclusive_jets(0.0)) {
    const auto cons = jet.constituents();
    if (int(cons.size()) < nRechitMin_)
      continue;
    const int ic = eta.size();
    double sx = 0, sy = 0, sz = 0, st = 0, st2 = 0, sc = 0, sd = 0;
    int nc = 0, nd = 0, me11 = 0, me12 = 0, me41 = 0, me42 = 0, mb1 = 0, mb2 = 0;
    std::map<int, int> perStation;  // key: 10*isDT + station
    for (const auto& c : cons) {
      const Hit& h = hits[c.user_index()];
      sx += h.x;
      sy += h.y;
      sz += h.z;
      st += h.t;
      st2 += double(h.t) * h.t;
      perStation[10 * h.dt + h.station]++;
      if (h.dt) {
        nd++;
        sd += h.t;
        mb1 += h.station == 1;
        mb2 += h.station == 2;
      } else {
        nc++;
        sc += h.t;
        me11 += h.station == 1 && h.ring == 1;
        me12 += h.station == 1 && h.ring == 2;
        me41 += h.station == 4 && h.ring == 1;
        me42 += h.station == 4 && h.ring == 2;
      }
      if (saveHits_) {
        hCl.push_back(ic);
        hx.push_back(h.x);
        hy.push_back(h.y);
        hz.push_back(h.z);
        ht.push_back(h.t);
        he.push_back(h.e);
        hDt.push_back(h.dt);
        hSt.push_back(h.station);
        hRing.push_back(h.ring);
      }
    }
    const double n = cons.size();
    int ns = 0, tot = 0;
    double avg = 0;
    for (const auto& [k, cnt] : perStation)
      if (cnt >= nStationThres_) {
        ns++;
        avg += (k % 10) * cnt;
        tot += cnt;
      }
    const double cx = sx / n, cy = sy / n, cz = sz / n, cr = std::hypot(cx, cy), tm = st / n;
    eta.push_back(std::asinh(cz / cr));
    phi.push_back(std::atan2(cy, cx));
    x.push_back(cx);
    y.push_back(cy);
    z.push_back(cz);
    r.push_back(cr);
    size.push_back(n);
    nCsc.push_back(nc);
    nDt.push_back(nd);
    nSt.push_back(ns);
    avgSt.push_back(tot ? avg / tot : 0.);
    nME11.push_back(me11);
    nME12.push_back(me12);
    nME41.push_back(me41);
    nME42.push_back(me42);
    nMB1.push_back(mb1);
    nMB2.push_back(mb2);
    time.push_back(tm);
    tSpread.push_back(std::sqrt(std::max(0., st2 / n - tm * tm)));
    tCsc.push_back(nc ? sc / nc : -999.);
    tDt.push_back(nd ? sd / nd : -999.);
  }

  auto t = std::make_unique<nanoaod::FlatTable>(eta.size(), name_, false, false);
  t->setDoc("joint CSC+DT shower clusters (CA R=" + std::to_string(rParam_) + ", >= " + std::to_string(nRechitMin_) +
            " hits); built for ML hits, time = TOF - OFF ns, prompt muon = 0");
  t->addColumn<float>("eta", eta, "cluster eta (hit-mean position)");
  t->addColumn<float>("phi", phi, "cluster phi");
  t->addColumn<float>("x", x, "mean hit x (cm)");
  t->addColumn<float>("y", y, "mean hit y (cm)");
  t->addColumn<float>("z", z, "mean hit z (cm)");
  t->addColumn<float>("r", r, "transverse radius of the mean hit position (cm)");
  t->addColumn<int>("size", size, "hits in the cluster (CSC + DT)");
  t->addColumn<int>("nCsc", nCsc, "CSC hits");
  t->addColumn<int>("nDt", nDt, "DT hits");
  t->addColumn<int>("nStation", nSt, "stations (CSC and DT counted apart) with >= nStationThres hits");
  t->addColumn<float>("avgStation", avgSt, "hit-weighted mean station over those stations");
  t->addColumn<int>("nME11", nME11, "ME1/1 hits (ME1/a + ME1/1b)");
  t->addColumn<int>("nME12", nME12, "ME1/2 hits");
  t->addColumn<int>("nME41", nME41, "ME4/1 hits");
  t->addColumn<int>("nME42", nME42, "ME4/2 hits");
  t->addColumn<int>("nMB1", nMB1, "MB1 hits");
  t->addColumn<int>("nMB2", nMB2, "MB2 hits");
  t->addColumn<float>("time", time, "mean hit time, CSC and DT (ns)");
  t->addColumn<float>("timeSpread", tSpread, "rms hit time (ns)");
  t->addColumn<float>("timeCsc", tCsc, "mean CSC hit time (ns), -999 if none");
  t->addColumn<float>("timeDt", tDt, "mean DT hit time (ns), -999 if none");
  ev.put(std::move(t), name_);

  if (saveHits_) {
    auto h = std::make_unique<nanoaod::FlatTable>(hCl.size(), name_ + "Hits", false, false);
    h->addColumn<int>("clusterIdx", hCl, "index into " + name_);
    h->addColumn<float>("x", hx, "global x (cm)");
    h->addColumn<float>("y", hy, "global y (cm)");
    h->addColumn<float>("z", hz, "global z (cm)");
    h->addColumn<float>("t", ht, "time (ns)");
    h->addColumn<float>("e", he, "energy (GeV); -1 for DT hits without hit info");
    h->addColumn<int>("isDt", hDt, "1 = DT, 0 = CSC");
    h->addColumn<int>("station", hSt, "station");
    h->addColumn<int>("ring", hRing, "CSC ring (ME1/a -> 1) or DT wheel");
    ev.put(std::move(h), name_ + "Hits");
  }
}

DEFINE_FWK_MODULE(muonMDSClusterTableProducer);
