# This package
using RandomSearch

# stdlib
using Logging, Test

# JSO
using NLPModels

function test()
    @testset "Solution is starting point" begin
        nlp = ADNLPModel(x -> x[1]^2 + x[2]^2, zeros(2))
        output = with_logger(NullLogger()) do
            random_search(nlp)
        end
        @test output.solution == zeros(2)
        @test output.objective == 0
    end
end

test()
