#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr int W=10, H=20, MAX_ACTIONS=64, E=MAX_ACTIONS, I=8, N=E+I;
using Board=std::array<std::array<int,W>,H>;
struct Cell { int x,y; };
using Shape=std::vector<Cell>;
const std::array<Shape,7> BASE{{
 {{0,0},{1,0},{2,0},{3,0}},
 {{0,0},{1,0},{0,1},{1,1}},
 {{1,0},{0,1},{1,1},{2,1}},
 {{1,0},{2,0},{0,1},{1,1}},
 {{0,0},{1,0},{1,1},{2,1}},
 {{0,0},{0,1},{1,1},{2,1}},
 {{2,0},{0,1},{1,1},{2,1}}
}};
struct Candidate { int rot=0,x=0,y=0,lines=0,holes=0,height=0,bump=0,maxh=0; double utility=0,q=0; std::array<double,5> phi{}; Board board{}; };
struct GameResult { std::string agent; unsigned seed=0; int score=0,lines=0,pieces=0,decisions=0; std::uint64_t neural_active_decisions=0, neural_action_spikes=0; bool topout=false; };

Shape rotate_shape(const Shape& s) {
 Shape r; for(auto c:s) r.push_back({c.y,-c.x});
 int minx=99,miny=99; for(auto c:r){minx=std::min(minx,c.x);miny=std::min(miny,c.y);}
 for(auto& c:r){c.x-=minx;c.y-=miny;} return r;
}
std::vector<Shape> rotations(const Shape& s) {
 std::vector<Shape> out; Shape cur=s;
 for(int k=0;k<4;++k){ bool seen=false; for(const auto& old:out){if(old.size()!=cur.size())continue; bool same=true; for(auto c:cur) if(std::none_of(old.begin(),old.end(),[&](Cell q){return q.x==c.x&&q.y==c.y;})){same=false;break;} if(same){seen=true;break;}} if(!seen)out.push_back(cur); cur=rotate_shape(cur);}
 return out;
}
bool collides(const Board& b,const Shape& s,int x,int y){
 for(auto c:s){int xx=x+c.x,yy=y+c.y;if(xx<0||xx>=W||yy>=H)return true;if(yy>=0&&b[yy][xx])return true;}return false;
}
Board lock_piece(Board b,const Shape& s,int x,int y,int& cleared){
 for(auto c:s){int yy=y+c.y,xx=x+c.x;if(yy>=0&&yy<H)b[yy][xx]=1;}
 int write=H-1;cleared=0;
 for(int read=H-1;read>=0;--read){bool full=std::all_of(b[read].begin(),b[read].end(),[](int v){return v!=0;});if(full){++cleared;continue;}b[write--]=b[read];}
 while(write>=0)b[write--].fill(0);
 return b;
}
void features(Candidate& c){
 std::array<int,W> heights{}; c.holes=0;c.height=0;c.maxh=0;
 for(int x=0;x<W;++x){bool found=false;for(int y=0;y<H;++y){if(c.board[y][x]&&!found){heights[x]=H-y;found=true;}else if(!c.board[y][x]&&found)++c.holes;}c.height+=heights[x];c.maxh=std::max(c.maxh,heights[x]);}
 c.bump=0;for(int x=1;x<W;++x)c.bump+=std::abs(heights[x]-heights[x-1]);
 c.utility=0.76*c.lines-0.51*c.holes-0.18*c.height-0.25*c.bump-0.35*c.maxh;
 c.phi={double(c.lines),double(c.holes)/10.0,double(c.height)/100.0,double(c.bump)/30.0,double(c.maxh)/20.0};
}
std::vector<Candidate> legal_moves(const Board& b,const Shape& piece){
 std::vector<Candidate> out;
 for(int r=0;r<(int)rotations(piece).size();++r){const auto rs=rotations(piece)[r];int maxx=0;for(auto c:rs)maxx=std::max(maxx,c.x);
  for(int x=0;x+maxx<W;++x){int y=-4;if(collides(b,rs,x,y))continue;while(!collides(b,rs,x,y+1))++y;if(y<0)continue;
   int lines=0;Candidate c;c.rot=r;c.x=x;c.y=y;c.board=lock_piece(b,rs,x,y,lines);c.lines=lines;features(c);out.push_back(c);}
 }
 return out;
}
CSRTopology make_brain_topology(){
 CSRTopology g;g.excitatory_count=E;g.inhibitory_count=I;g.minimum_delay=1;g.maximum_delay=1;
 g.row_offsets.assign(N+1,0);
 for(int s=0;s<N;++s){if(s<E){for(int t=E;t<N;++t){g.target_indices.push_back(t);g.weights.push_back(2.0f);g.delays.push_back(1);}}
 else{for(int t=0;t<E;++t){g.target_indices.push_back(t);g.weights.push_back(-1.5f);g.delays.push_back(1);}}
 g.row_offsets[s+1]=(std::uint32_t)g.weights.size();}
 g.validate();return g;
}
class NeuralChooser {
 public:
 explicit NeuralChooser(const std::filesystem::path& dir,unsigned seed):path_(dir/"tetris_brain.csr"){
  save_csr_topology(make_brain_topology(),path_.string());
  ModelParameters p;p.simulation.excitatory_count=E;p.simulation.inhibitory_count=I;p.simulation.steps=1000000;p.simulation.dt=0.5f;p.simulation.threshold=30.0f;p.simulation.random_seed=seed;
  p.simulation.initial_voltage_min=-65.0f;p.simulation.initial_voltage_max=-65.0f;
  p.simulation.excitatory_current_min=0;p.simulation.excitatory_current_max=0;p.simulation.inhibitory_current_min=0;p.simulation.inhibitory_current_max=0;
  p.topology.p_ee=p.topology.p_ei=p.topology.p_ie=p.topology.p_ii=0;p.topology.master_seed=seed;p.topology.topology_seed=seed;p.topology.delay_seed=seed+1;
  runtime_=std::make_unique<CudaSimulationRuntime>(p,path_.string(),0,true,false);
 }
 int choose(const std::vector<Candidate>& moves){
  if(moves.empty())return -1;
  runtime_->clear_stimuli();
  // The runtime requires external pulses to begin strictly after the completed step.
  const auto start=runtime_->completed_steps()+1U;
  for(std::size_t i=0;i<moves.size()&&i<MAX_ACTIONS;++i){
   const double u=std::tanh(moves[i].q/60.0);
   const float current=(float)std::clamp(300.0+200.0*u,80.0,500.0);
   runtime_->schedule_stimulus((std::uint32_t)i,start,4,current);
  }
  runtime_->run_steps(60);
  const auto events=runtime_->collect_spike_events();
  std::array<int,MAX_ACTIONS> counts{};
  for(const auto& ev:events.events)if(ev.step>=start&&ev.neuron_id<MAX_ACTIONS)++counts[ev.neuron_id];
  std::uint64_t total=0;for(std::size_t i=0;i<moves.size()&&i<MAX_ACTIONS;++i)total+=counts[i];
  total_action_spikes_+=total;if(total>0)++active_decisions_;
  int best=0;for(std::size_t i=1;i<moves.size()&&i<MAX_ACTIONS;++i){
   if(counts[i]>counts[best]||(counts[i]==counts[best]&&moves[i].q>moves[best].q))best=(int)i;
  }
  return best;
 }
 std::uint64_t active_decisions() const{return active_decisions_;}
 std::uint64_t action_spikes() const{return total_action_spikes_;}
 private:
 std::filesystem::path path_;std::unique_ptr<CudaSimulationRuntime> runtime_;
 std::uint64_t active_decisions_=0,total_action_spikes_=0;
};
class LinearQPolicy {
 public:
  explicit LinearQPolicy(double alpha=0.002,double gamma=0.95):alpha_(alpha),gamma_(gamma){}
  double value(const Candidate& c) const {
   double q=0.0;for(std::size_t i=0;i<w_.size();++i)q+=w_[i]*c.phi[i];return q;
  }
  void update(const Candidate& c,double reward,double next_max,bool terminal){
   const double target=reward+(terminal?0.0:gamma_*next_max);
   const double delta=target-value(c);
   for(std::size_t i=0;i<w_.size();++i)w_[i]=std::clamp(w_[i]+alpha_*delta*c.phi[i],-500.0,500.0);
   ++updates_;
  }
  const std::array<double,5>& weights() const{return w_;}
  std::uint64_t updates() const{return updates_;}
 private:
  std::array<double,5> w_{0.0,0.0,0.0,0.0,0.0};
  double alpha_,gamma_;
  std::uint64_t updates_=0;
};
GameResult play(const std::string& agent,unsigned seed,int max_pieces,const std::filesystem::path& outdir,
                LinearQPolicy* policy=nullptr,bool training=false){
 GameResult result;result.agent=agent;result.seed=seed;
 std::mt19937 piece_rng(seed),action_rng(seed^0x9e3779b9U);
 std::uniform_int_distribution<int> piece_dist(0,6);
 std::vector<int> sequence(max_pieces);for(auto& p:sequence)p=piece_dist(piece_rng);
 std::uniform_real_distribution<double> unit
(0.0,1.0);
 Board board{};std::unique_ptr<NeuralChooser> brain;
 if(agent=="neuromancer")brain=std::make_unique<NeuralChooser>(outdir,seed);
 static const int points[5]={0,100,300,500,800};
 for(int turn=0;turn<max_pieces;++turn){
  const Shape& piece=BASE[sequence[turn]];
  auto moves=legal_moves(board,piece);
  if(moves.empty()){result.topout=true;break;}
  for(auto& c:moves)c.q=policy?policy->value(c):c.utility;
  int chosen=0;
  if(agent=="random"){
   std::uniform_int_distribution<int>d(0,(int)moves.size()-1);chosen=d(action_rng);
  }else if(agent=="heuristic"){
   for(int i=1;i<(int)moves.size();++i)if(moves[i].utility>moves[chosen].utility)chosen=i;
  }else if(agent=="neuromancer"){
   chosen=brain->choose(moves);if(chosen<0)chosen=0;
   // Epsilon exploration is used only while learning; evaluation is fully greedy.
   if(training&&unit(action_rng)<0.15){
    std::uniform_int_distribution<int>d(0,(int)moves.size()-1);chosen=d(action_rng);
   }
  }else throw std::runtime_error("unknown agent");
  const Candidate selected=moves[chosen];
  board=selected.board;result.lines+=selected.lines;
  result.score+=points[std::clamp(selected.lines,0,4)];
  ++result.pieces;++result.decisions;
  if(policy&&training){
   auto next_moves=(turn+1<max_pieces)?legal_moves(board,BASE[sequence[turn+1]]):std::vector<Candidate>{};
   const bool topout=next_moves.empty()&&turn+1<max_pieces;
   const bool terminal=topout||turn+1==max_pieces;
   double next_max=0.0;
   for(const auto& n:next_moves)next_max=std::max(next_max,policy->value(n));
   const double reward=1.0+points[std::clamp(selected.lines,0,4)]-(topout?100.0:0.0);
   policy->update(selected,reward,next_max,terminal);
   if(topout){result.topout=true;break;}
  }
 }
 if(brain){result.neural_active_decisions=brain->active_decisions();result.neural_action_spikes=brain->action_spikes();}
 return result;
}
} // namespace

int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: neuromancer_tetris_v38 <output-directory>");
  std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
  std::ofstream csv(out/"tetris_v38_games.csv");if(!csv)throw std::runtime_error("cannot open results CSV");
  csv<<"stage,agent,seed,score,lines,pieces,decisions,neural_active_decisions,neural_action_spikes,topout\n";
  LinearQPolicy policy;
  constexpr int training_episodes=24,training_piece_limit=350;
  for(int ep=0;ep<training_episodes;++ep){
   const unsigned seed=38000U+(unsigned)ep;
   auto r=play("neuromancer",seed,training_piece_limit,out,&policy,true);
   csv<<"train,"<<r.agent<<','<<r.seed<<','<<r.score<<','<<r.lines<<','<<r.pieces<<','<<r.decisions<<','<<r.neural_active_decisions<<','<<r.neural_action_spikes<<','<<(r.topout?1:0)<<'\n';csv.flush();
   std::cout<<"train episode="<<(ep+1)<<"/"<<training_episodes<<" score="<<r.score<<" lines="<<r.lines<<" pieces="<<r.pieces<<" updates="<<policy.updates()<<'\n';
  }
  std::ofstream wf(out/"tetris_v38_learned_weights.csv");
  wf<<"feature,weight\n";
  const std::array<const char*,5> names{"lines","holes_per_10","aggregate_height_per_100","bumpiness_per_30","max_height_per_20"};
  for(std::size_t i=0;i<names.size();++i)wf<<names[i]<<','<<std::setprecision(12)<<policy.weights()[i]<<'\n';
  constexpr std::array<unsigned,8> eval_seeds{38101,38202,38303,38404,38505,38606,38707,38808};
  for(auto seed:eval_seeds)for(const std::string agent:{"random","heuristic","neuromancer"}){
   auto r=play(agent,seed,500,out,&policy,false);
   csv<<"evaluation,"<<r.agent<<','<<r.seed<<','<<r.score<<','<<r.lines<<','<<r.pieces<<','<<r.decisions<<','<<r.neural_active_decisions<<','<<r.neural_action_spikes<<','<<(r.topout?1:0)<<'\n';csv.flush();
   std::cout<<agent<<" seed="<<seed<<" score="<<r.score<<" lines="<<r.lines<<" pieces="<<r.pieces<<" topout="<<r.topout<<'\n';
  }
  std::cout<<"NEUROMANCER V38 — REWARD-DRIVEN TETRIS POLICY\n"
           <<"Training: "<<training_episodes<<" episodes, epsilon=0.15, TD(0) linear Q policy.\n"
           <<"Evaluation: 8 held-out matched piece-sequence seeds; exploration disabled and weights frozen.\n"
           <<"Neural WTA selects among actions using learned Q values; Q weights update from line-clear, survival, and top-out rewards.\n"
           <<"This is reward-driven linear function approximation coupled to the spiking controller, not synaptic STDP learning.\n"
           <<"Wrote "<<(out/"tetris_v38_games.csv").string()<<'\n';
  return 0;
 }catch(const std::exception&e){std::cerr<<"V38 Tetris failed: "<<e.what()<<'\n';return 1;}
}
