is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

function cuttemplate(data, sensorscoordinates, template, starttime, freq, speed, window)
    templatecoordinates = [template.north, template.east, template.up]
    originsample = round(Int, 1e-3freq * (template.time - starttime))
    head_len, tail_len = window
    templatedata = similar(data)
    templateoffsets = Offsets(undef, length(data))
    for channel in eachindex(data)
        displacement = sensorscoordinates[channel] .- templatecoordinates
        distance = Base.splat(hypot)(displacement)
        offset = round(Int, freq * (distance / speed))
        arrivalsample = originsample + offset
        templateoffsets[channel] = offset
        templatedata[channel] = data[channel][arrivalsample - head_len: arrivalsample + tail_len]
    end
    templatedata, templateoffsets
end