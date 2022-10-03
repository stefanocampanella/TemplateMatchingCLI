function readlabfile(filepath, nb, ::Type{T})::Vector{T} where {T <: AbstractFloat} 
    ntoh.(reinterpret(Int16, read(filepath, nb)))
end

function readlabdir(dirpath, datetime, experiment, num_channels, nb, eltype::Type{T}) where T <: AbstractFloat
    data = Vector{Vector{eltype}}(undef, num_channels)
    for n = 0:2:num_channels - 1
        filepath = joinpath(dirpath, @sprintf "%s_%s_ch%d&%d.bin" datetime experiment n (n + 1))
        interleaved_data = readlabfile(filepath, nb, eltype)
        data[n + 1] = interleaved_data[1:2:end]
        data[n + 2] = interleaved_data[2:2:end]
    end
    data
end

function readcatalogue(filepath, origintime)
    df = CSV.read(filepath, DataFrame)
    sec = floor.(Int, df.Second)
    datetimes = DateTime.(df.Year, df.Month, df.Day, df.Hour, df.Minute, sec)
    catalogue = DataFrame()
    catalogue.index = 1:nrow(df)
    catalogue.north = df.North
    catalogue.east = df.East
    catalogue.up = df.Up
    catalogue.time = map(x -> 1e3(x - origintime).value, datetimes) .+ 1e6(df.Second - sec)
    catalogue.magnitude = df.magnitude
    catalogue.weights = map(Vector, eachrow(df[!, filter(startswith("wp"), sort(names(df), lt=natural))]))
    catalogue
end

readsensorscoordinates(filepath; header=[:north, :east, :up]) = Vector.(eachrow(CSV.read(filepath, DataFrame; header)))