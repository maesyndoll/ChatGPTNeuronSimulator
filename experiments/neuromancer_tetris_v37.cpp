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
struct Candidate { int rot=0,x=0,y=0,lines=0,holes=0,height=0,bump=0,maxh=0; double utility=0; Board board{}; };
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
   const double u=std::clamp(moves[i].utility,-25.0,5.0);
   const float current=(float)std::clamp(360.0+u*12.0,80.0,500.0);
   runtime_->schedule_stimulus((std::uint32_t)i,start,4,current);
  }
  runtime_->run_steps(60);
  const auto events=runtime_->collect_spike_events();
  std::array<int,MAX_ACTIONS> counts{};
  for(const auto& ev:events.events)if(ev.step>=start&&ev.neuron_id<MAX_ACTIONS)++counts[ev.neuron_id];
  std::uint64_t total=0;for(std::size_t i=0;i<moves.size()&&i<MAX_ACTIONS;++i)total+=counts[i];
  total_action_spikes_+=total;if(total>0)++active_decisions_;
  int best=0;for(std::size_t i=1;i<moves.size()&&i<MAX_ACTIONS;++i){
   if(counts[i]>counts[best]||(counts[i]==counts[best]&&moves[i].utility>moves[best].utility))best=(int)i;
  }
  return best;
 }
 std::uint64_t active_decisions() const{return active_decisions_;}
 std::uint64_t action_spikes() const{return total_action_spikes_;}
 private:
 std::filesystem::path path_;std::unique_ptr<CudaSimulationRuntime> runtime_;
 std::uint64_t active_decisions_=0,total_action_spikes_=0;
};
GameResult play(const std::string& agent,unsigned seed,int max_pieces,const std::filesystem::path& outdir){
 GameResult result;result.agent=agent;result.seed=seed;std::mt19937 rng(seed);std::uniform_int_distribution<int> piece_dist(0,6);
 Board board{};std::unique_ptr<NeuralChooser> brain;if(agent=="neuromancer")brain=std::make_unique<NeuralChooser>(outdir,seed);
 for(int turn=0;turn<max_pieces;++turn){const Shape& piece=BASE[piece_dist(rng)];auto moves=legal_moves(board,piece);if(moves.empty()){result.topout=true;break;}
  int chosen=0;
  if(agent=="random"){std::uniform_int_distribution<int>d(0,(int)moves.size()-1);chosen=d(rng);}
  else if(agent=="heuristic"){for(int i=1;i<(int)moves.size();++i)if(moves[i].utility>moves[chosen].utility)chosen=i;}
  else if(agent=="neuromancer"){chosen=brain->choose(moves);if(chosen<0)chosen=0;}
  else throw std::runtime_error("unknown agent");
  board=moves[chosen].board;result.lines+=moves[chosen].lines;
  static const int points[5]={0,100,300,500,800};result.score+=points[std::clamp(moves[chosen].lines,0,4)];
  ++result.pieces;++result.decisions;
 }
 if(brain){result.neural_active_decisions=brain->active_decisions();result.neural_action_spikes=brain->action_spikes();}
 return result;
}
}
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: neuromancer_tetris_v37 <output-directory>");
  std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
  std::ofstream csv(out/"tetris_v37_games.csv");if(!csv)throw std::runtime_error("cannot open results CSV");
  csv<<"agent,seed,score,lines,pieces,decisions,neural_active_decisions,neural_action_spikes,topout\n";
  constexpr std::array<unsigned,4> seeds{37031,37132,37233,37334};
  for(auto seed:seeds)for(const std::string agent:{"random","heuristic","neuromancer"}){
   auto r=play(agent,seed,500,out);csv<<r.agent<<','<<r.seed<<','<<r.score<<','<<r.lines<<','<<r.pieces<<','<<r.decisions<<','<<r.neural_active_decisions<<','<<r.neural_action_spikes<<','<<(r.topout?1:0)<<'\n';csv.flush();
   std::cout<<agent<<" seed="<<seed<<" score="<<r.score<<" lines="<<r.lines<<" pieces="<<r.pieces<<" topout="<<r.topout<<'\n';
  }
  std::cout<<"NEUROMANCER V37 — CLOSED-LOOP TETRIS PROTOTYPE\n"
           <<"Games: 12 (4 matched piece-sequence seeds x 3 agents)\n"
           <<"Agents: random, hand-scored placement baseline, CUDA Neuromancer WTA controller.\n"
           <<"Neuromancer receives candidate-placement utility as graded external-current input; recurrent E/I competition selects an action.\n"
           <<"This is a hybrid neural decision prototype, not learned Tetris policy; no reward-driven weight learning yet.\n"
           <<"Wrote "<<(out/"tetris_v37_games.csv").string()<<'\n';
  return 0;
 }catch(const std::exception&e){std::cerr<<"V37 Tetris failed: "<<e.what()<<'\n';return 1;}
}