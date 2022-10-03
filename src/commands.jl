"""
Read binary files, de-trend, filter, resample, and finally save data on disk. 
Binary files should be named <`datetime`>_<`experiment``>_ch<NN>&<MM>.bin and located 
all in the same directory `inputdirpath`.

# Arguments

- `inputdirpath`: directory of the binary files
- `datastartstr`: datetime of data to be read
- `experiment`: experiment identifier of data to be read
- `experimentstartstr`: time of experiment initiation
- `badsamplespath`: path of bad samples CSV
- `num_channels`: number of channels
- `outputdirpath`: path of the output directory

# Options

- `-i, --inputfreq`: input frequency in KHz 
- `-o, --outputfreq`: output frequency in KHz 
- `-l, --lopassfreq`: lower frequency in bandpass filter in KHz
- `-h, --hipassfreq`: higher frequency in bandpass filter in KHz
- `-n, --numpoles`: number of poles in Butterworth filter
"""
@cast function preprocess(dirpath::AbstractString, datastartstr::AbstractString, 
                          experiment::AbstractString, experimentstartstr::AbstractString, 
                          badsamplespath::AbstractString,
                          num_channels::Int,
                          outputpath::AbstractString;
                          inputfreq::Int=10_000, outputfreq::Int=1000,
                          lopassfreq::Int=50, hipassfreq::Int=400, numpoles::Int=4)
    datastart = DateTime(datastartstr, dateformat"yyyy-mm-dd_HH-MM-SS")
    experimentstart = DateTime(experimentstartstr, dateformat"yyyy-mm-dd_HH-MM-SS")
    @info "Reading files from $dirpath (experiment: $experiment (starting at $experimentstart), time: $datastart, nch: $num_channels)"
    data = readlabdir(dirpath, datastartstr, experiment, num_channels, typemax(Int), Float32)
    @info "Reading bad samples CSV at $badsamplespath"
    badsamplesranges = UnitRange{Int}[]
    badsamplesdf = CSV.read(badsamplespath, DataFrame)
    filter!(:datetime => ==(datastart), badsamplesdf)
    for row in eachrow(badsamplesdf)
        push!(badsamplesranges, row.startsample:row.endsample)
    end
    @info "Pre-processing data"
    responsetype = Bandpass(lopassfreq, hipassfreq, fs=inputfreq)
    designmethod = Butterworth(numpoles)
    progressbar = Progress(length(data); output=stderr, enabled=!is_logging(stderr))
    Threads.@threads for n in eachindex(data)
        ys = data[n]
        for r in badsamplesranges
            ys[r] .= 0.0
        end
        xs = axes(data[n], 1)
        beta = (mean(xs .* ys) - mean(xs) * mean(ys)) / std(xs, corrected=false)
        alpha = mean(ys) - beta * mean(xs)
        ys_detrended = @. ys - alpha - beta * xs
        ys_filtered = filtfilt(digitalfilter(responsetype, designmethod), ys_detrended)
        ys_resampled = resample(ys_filtered, outputfreq // inputfreq)
        data[n] = ys_resampled
        next!(progressbar)
    end
    datastart_us = 1e3(datastart - experimentstart).value
    dataend_us = datastart_us + 1e3length(first(data)) / outputfreq
    @info "Saving data at $outputpath"
    jldsave(outputpath; data, datastart=datastart_us, dataend=dataend_us, freq=outputfreq)
end

"""
Cut templates.

# Arguments

- `datadirpath`: path of the directory of JLD2 data files
- `sensorspath`: path of the CSV containing sensors coordinates in cm
- `cataloguepath`: path of the CSV catalogue of templates
- `experimentstartstr`: time of experiment initiation
- `outputpath`: path of the output file

# Options

- `-s, --speed`: P-wave speed in cm/ms
- `-w, --window`: template window in samples
"""
@cast function maketemplates(datadirpath::AbstractString, sensorspath::AbstractString, 
                             cataloguepath::AbstractString, experimentstartstr::AbstractString,
                             outputpath::AbstractString; 
                             speed::Float64=670.0, window::Tuple{Int, Int}=(50, 250))
    experimentstart = DateTime(experimentstartstr, dateformat"yyyy-mm-dd_HH-MM-SS")
    @info "Reading catalogue from $cataloguepath"
    catalogue = readcatalogue(cataloguepath, experimentstart)
    @info "Reading sensors coordinates from $sensorspath"
    sensorscoordinates = readsensorscoordinates(sensorspath)
    @info "Reading data and cutting templates"
    templates_data = Vector{MaybeStream{Float32}}(missing, nrow(catalogue))
    templates_offsets = Vector{MaybeOffsets}(missing, nrow(catalogue))
    datapaths = filter(endswith(".jld2"), readdir(datadirpath, join=true))
    progressbar = Progress(length(datapaths); output=stderr, enabled=!is_logging(stderr))
    Threads.@threads for datapath in datapaths
        data, datastart, dataend, freq = load(datapath, "data", "datastart", "dataend", "freq")
        templates_within_data = filter(row -> datastart <= row.time < dataend, catalogue)
        for template in eachrow(templates_within_data)
            try
                template_data, offsets = cuttemplate(data, sensorscoordinates, template, datastart, freq, speed, window)
                templates_data[template.index] = template_data
                templates_offsets[template.index] = offsets
            catch error
                @warn "An exception occurred while preparing a template. Skipping." template error
            end
        end
        next!(progressbar)
    end
    @info "Saving templates"
    catalogue.data = templates_data
    catalogue.offsets = templates_offsets
    jldsave(outputpath; catalogue, speed, window, sensorscoordinates)
end

"""
Match templates.

# Arguments

- `datapath`: path of continuous data
- `templatespath`: path of the directory of JLD2 data files
- `outputpath`: path of the output file

# Options

- `--tolerance`: sample tolerance in stacking
- `--threshold`: height threshold
- `--reldistance`: minimum distance between peaks
"""
@cast function matchtemplates(datapath::AbstractString, templatespath::AbstractString, 
                              outputpath::AbstractString; 
                              tolerance::Int=8, threshold::Float32=0.4f0, reldistance::Int=2)
    @info "Reading data from $datapath"
    data, datastart = load(datapath, "data", "datastart")
    @info "Reading templates from $templatespath"
    catalogue, window = load(templatespath, "catalogue", "window")
    @info "Computing cross-correlations and processing matches"
    if CUDA.functional()
        @info "GPU acceleration available" CUDA.version()
        iscudafunctional = true
        data_xpu = CuArray.(data)
    else
        @info "GPU acceleration not available"
        iscudafunctional = false
        data_xpu = data
    end
    templates = Tables.namedtupleiterator(catalogue)
    progressbar = Progress(length(templates); output=stderr, enabled=!is_logging(stderr), showspeed=true)
    peaks_chnl = Channel{Tuple{eltype(templates), Vector{Int}}}() do chnl
        for template in templates
            if !any(map(ismissing, template))
                templatedata = iscudafunctional ? CuArray.(template.data) : template.data
                signal = convert(OffsetVector{Float32, Vector{Float32}}, 
                                 correlatetemplate(data_xpu, templatedata, template.offsets, tolerance, Float32; usefft=true))
                peaks, _ = findpeaks(signal, threshold, reldistance * length(first(template.data)))
                if !isempty(peaks)
                    put!(chnl, (template, peaks))
                end
            end
            next!(progressbar)
        end
    end
    detections_chnl = Channel{DataFrame}() do chnl
        for (template, peaks) in peaks_chnl
            detections = DataFrame()
            detections.peak_sample = peaks .+ window[1]
            detections.template .= template.index
            details = map(p -> processdetection(data, template, p, window, tolerance), peaks)
            detections = hcat(detections, DataFrame(details))
            filter!(:correlations => ccs -> mean(ccs) >= threshold, detections)
            if !isempty(detections)
                put!(chnl, detections)
            end
        end
    end
    detections = collect(detections_chnl)
    if isempty(detections)
        @info "No match found."
    else
        augmented_catalogue = reduce(vcat, detections)
        @info "Found $(nrow(augmented_catalogue)) matches."
        @info "Saving augmented catalogue at $outputpath." augmented_catalogue
        jldsave(outputpath; augmented_catalogue, datastart)
    end
end