function processdetection(data::Stream{T}, template, peak, window, tolerance) where T <: AbstractFloat
    correlations = similar(data, Float64)
    arrivals = similar(data, Float64)
    waveforms = similar(data, Union{Missing, Vector{T}})
    Threads.@threads for ch in eachindex(data) 
        toa, cc = estimatetoa(data[ch], template.data[ch], peak + template.offsets[ch], tolerance)
        correlations[ch] = cc
        arrivals[ch] = toa
        waveforms[ch] = waveformat(data[ch], window, round(Int, toa))
    end
    (; arrivals, correlations, data=waveforms)
end


function waveformat(trace, window, sample)
    head_len, tail_len = window
    if firstindex(trace) + head_len <= sample <= lastindex(trace) - tail_len
        view(trace, sample - head_len:sample + tail_len)
    else
        missing
    end
end