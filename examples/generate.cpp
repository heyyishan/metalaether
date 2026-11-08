#include "aether/engine.hpp"
#include <iostream>
int main(int argc,char**argv){try{aether::Model model;if(argc>1)model=aether::Model::load(argv[1]);else{aether::Config c;c.dim=128;c.hidden_dim=384;c.n_layers=2;c.n_heads=4;c.max_seq_len=256;model=aether::Model::random_for_demo(c);model.save("demo.aether");}aether::Engine engine(std::move(model));std::cout<<"Device: "<<engine.device()<<"\nPrompt: ";std::string p;std::getline(std::cin,p);std::cout<<engine.generate(p,{64,0,1})<<"\n";}catch(const std::exception&e){std::cerr<<"aether: "<<e.what()<<"\n";return 1;}}
