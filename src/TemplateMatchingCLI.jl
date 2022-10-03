module TemplateMatchingCLI

using CSV
using CUDA
using Comonicon
using DSP
using DataFrames
using Dates
using JLD2
using LinearAlgebra
using NaturalSort
using OffsetArrays
using Printf
using ProgressMeter
using StatsBase
using Tables
using TemplateMatching


include("types.jl")
include("readers.jl")
include("process.jl")
include("utils.jl")
include("commands.jl")

@main

end
