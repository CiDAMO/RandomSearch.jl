module RandomSearch

export random_search

using NLPModels, SolverTools

function random_search(nlp :: AbstractNLPModel;
                       max_eval :: Int = 1000,
                       )
    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    fx = obj(nlp, x)

    @info log_header([:nf, :fx], [Int, Float64],
        hdr_override=Dict(:fx => "f(x)", :nf => "#f"))
    @info log_row(Any[neval_obj(nlp), fx])
    tired = neval_obj(nlp) > max_eval
    while !tired
        d = randn(n)
        xt = x + d
        ft = obj(nlp, xt)
        if ft < fx
            x .= xt
            fx = ft
        end
        tired = neval_obj(nlp) > max_eval
        @info log_row(Any[neval_obj(nlp), fx])
    end

    status = :max_eval
    return GenericExecutionStats(status, nlp,
            solution=x, objective=fx)
end

end # module
