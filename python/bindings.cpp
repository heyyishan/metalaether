#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include "aether/engine.hpp"
namespace py=pybind11;
PYBIND11_MODULE(_aether,m){
 py::class_<aether::Config>(m,"Config").def(py::init<>()).def_readwrite("vocab_size",&aether::Config::vocab_size).def_readwrite("dim",&aether::Config::dim).def_readwrite("hidden_dim",&aether::Config::hidden_dim).def_readwrite("n_layers",&aether::Config::n_layers).def_readwrite("n_heads",&aether::Config::n_heads).def_readwrite("max_seq_len",&aether::Config::max_seq_len);
 py::class_<aether::GenerateOptions>(m,"GenerateOptions").def(py::init<>()).def_readwrite("max_new_tokens",&aether::GenerateOptions::max_new_tokens).def_readwrite("temperature",&aether::GenerateOptions::temperature).def_readwrite("top_k",&aether::GenerateOptions::top_k);
 py::class_<aether::Model>(m,"Model").def_static("load",&aether::Model::load).def_static("random_for_demo",&aether::Model::random_for_demo).def("save",&aether::Model::save);
 py::class_<aether::BenchResult>(m,"BenchResult").def_readonly("milliseconds",&aether::BenchResult::milliseconds).def_readonly("tokens_per_second",&aether::BenchResult::tokens_per_second).def_readonly("bytes_moved",&aether::BenchResult::bytes_moved);
 py::class_<aether::Engine>(m,"Engine").def(py::init<aether::Model>()).def("generate",&aether::Engine::generate,py::arg("prompt"),py::arg("options")=aether::GenerateOptions{}).def("benchmark",&aether::Engine::benchmark,py::arg("prompt_tokens"),py::arg("new_tokens"),py::arg("batch")=1).def_property_readonly("device",&aether::Engine::device).def("profile_report",&aether::Engine::profile_report);
}
