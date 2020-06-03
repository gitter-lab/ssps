#! /bin/bash

# PackageCompiler is more robust when 
# we force it to reinstall/recompile everything.
# Julia somehow maintains state in a way that makes
# it hard to compile things in a reproducible fashion.

rm -rf ~/.julia

rm -rf SSPSCompiled

julia -e "using Pkg; Pkg.activate(\"SSPS\"); Pkg.instantiate(); using PackageCompiler; create_app(\"SSPS\", \"SSPSCompiled\"; precompile_execution_file=\"SSPS/precompile_script.jl\")" 
