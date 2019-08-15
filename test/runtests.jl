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

    @testset "Simple problem" begin
        nlp = ADNLPModel(x -> x[1]^2 + x[2]^2, ones(2))
        output = with_logger(NullLogger()) do
            random_search(nlp)
        end
        @test output.objective < 2
    end
end

test()
