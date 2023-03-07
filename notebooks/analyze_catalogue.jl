### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f00d14ea-b8db-4bf2-bfdf-90a4a3cb3ccf
begin
    import Pkg
    Pkg.activate(".")
end

# ╔═╡ 8b4426b8-588b-49ad-83c5-a79ed698d704
using PlutoUI

# ╔═╡ 3c07c33b-7f91-4e2d-a27a-09a8924f328c
using DataFrames

# ╔═╡ b59dd2c0-4c3e-4c3d-a4d4-f8da9dcbe286
using JLD2

# ╔═╡ 76e2e659-c708-4a08-8263-def6c3dfa930
using Dates

# ╔═╡ 533d8e9e-40ff-4740-bbd9-8199cf0bcef6
using TemplateMatchingCLI

# ╔═╡ a7c96f5f-db00-48ac-8945-c268200fd868
using Optim

# ╔═╡ 89c57605-3aea-4335-adae-55c40c972d26
using TemplateMatching

# ╔═╡ f07ab63f-dfb4-493c-a40c-6b3363a858c3
using Plots

# ╔═╡ 356cc4c2-e565-4847-9286-cfc1f835654d
using StatsBase

# ╔═╡ 3ec5251a-2e3e-4344-bb1f-fcd79daf9714
using LinearAlgebra

# ╔═╡ 79f1487a-0dc8-483b-a100-ecd176b7dd92
using CSV

# ╔═╡ 57b35604-727b-4eac-bcbd-ea302e4c79a2
md"## Reading template matched catalogue"

# ╔═╡ 2fc4fdfe-81b2-45f1-9ed4-b8ad5bb86ee3
md"""Path of the catalogue JLD $(@bind cataloguepath TextField(default="../data/gabbro7/detections.jld2"))"""

# ╔═╡ 43f3ae0e-9eb0-47d1-b750-b04355f12082
candidates = load(cataloguepath, "detections")

# ╔═╡ 7987b27b-c7ae-4f97-93ee-b83971ec0248
md"## Read templates catalogue"

# ╔═╡ 6d399562-aed4-42a6-b4ee-f1b6cb2c9a4e
md"""Path of the templates JLD $(@bind templatespath TextField(default="../data/gabbro7/templates.jld2"))"""

# ╔═╡ fea071ff-179b-46b0-8bff-953712eb8fd8
templates = load(templatespath, "catalogue")

# ╔═╡ fd864a98-14ee-4bdd-9a8d-52e80ee4f5a1
let ts = dropmissing(templates)
	sort([n for n in 1:nrow(templates) if n ∉ Set(ts.index)])
end

# ╔═╡ 6ea0c487-57e6-4ade-bd45-15226df212ec
md"## Read sensors positions"

# ╔═╡ 2c1941d0-ab86-40cb-8897-d1898e41f27c
sensors = load(templatespath, "sensorscoordinates")

# ╔═╡ 102e4534-6be4-4436-be36-69726a393253
md" ## Catalogue cleaning "

# ╔═╡ 035a667d-e948-4dbe-8ae0-bcfd571a2da9
template_counts = combine(groupby(candidates, :template), nrow => :count)

# ╔═╡ 20ac88b1-5a8a-4919-b6cd-8b4a7d829349
bar(template_counts.template, template_counts.count, label=nothing)

# ╔═╡ 12950cbd-cfe8-4a53-a93c-7da9aec408bc
let
	N = nrow(templates)
	magnitudes = Vector{Float64}(undef, N)
	for (n, template) in enumerate(Tables.namedtupleiterator(dropmissing(templates)))
		ws = weights(template.weights)
		if any(ismissing, template.data)
			magnitudes[n] = missing
		else
			amps_squared = maximum.(map(series -> series .^ 2, template.data))
			x0 = [template.north, template.east, template.up]
			dists_squared = map(xs -> dot(xs - x0, xs - x0), sensors)
			mag = 0.5log10(mean(amps_squared .* dists_squared, ws)) - 2
			magnitudes[n] = mag
		end
	end
	templates.magnitude_recomputed = magnitudes
end

# ╔═╡ 81154535-0576-4a69-b906-dabc3d7c8f9d
let
	plt = plot()
	histogram!(plt, templates.magnitude_recomputed, label="recomputed magnitudes", fillalpha=0.5)
	histogram!(plt, templates.magnitude, label="catalogue magnitudes", fillalpha=0.5)
	plt
end

# ╔═╡ d4b0ffe4-92f8-4a25-a259-aa9175f39933
bad_templates = filter(:count => >(100), template_counts)

# ╔═╡ 9a1c9fe2-8f1d-4f2a-b0b4-ef1bf91affdc
empty_templates = filter(:magnitude_recomputed => <=(0.), templates)

# ╔═╡ aed075ff-6ca6-4dd6-bb64-ab75177f654e
good_templates = filter(:index => !in(union(bad_templates.template, empty_templates.index)), templates)

# ╔═╡ fe822a34-bdc7-486f-a225-2ed2c15a8371
detections = let 
	df = copy(candidates)
	filter!(:data => xs -> !any(ismissing.(xs)), df)
	filter!(:template => in(good_templates.index), df)
	filter!(:correlations => ccs -> count(>=(0.5), ccs) >= 8, df)
end

# ╔═╡ 26c5d8c7-ce56-41d6-bab5-63a0f1cc369e
md"## Localize detections"

# ╔═╡ 417f915e-0013-48b6-bc22-2b04125d4c18
speed = 0.615 # cm / us

# ╔═╡ b963608d-92a1-4416-9a34-bb60b63b6406
samples_pre = 50

# ╔═╡ 9205dd4a-c67e-470c-9bd1-06fa03187b34
samplestotimes = 1.0

# ╔═╡ 3d6696cc-be33-4c5a-99f5-15fa1317a308
line_element_squared(x, t) = (dot(x, x) - speed^2 * t^2)^2

# ╔═╡ 443cefe2-f349-4cd4-be3e-6b83537ae6df
function residue_rms(x0, t0, toas, ws)
    rs = map((x, t) -> line_element_squared(x - x0, t - t0), sensors, toas)
    sqrt(mean(rs, weights(ws)))
end

# ╔═╡ caaab003-d9be-41fa-99cf-d9464db0b612
locate(toas, x_guess, t_guess, ws) = optimize(xt -> residue_rms(xt[1:3], xt[4], toas, ws), [x_guess; t_guess])

# ╔═╡ 74e83a60-31a6-42ae-8367-c573c34483d5
let
	norths = Vector{Float64}(undef, 0)
	easts = Vector{Float64}(undef, 0)
	ups = Vector{Float64}(undef, 0)
	times = Vector{Float64}(undef, 0)
	residuals = Vector{Union{Float64, Missing}}(undef, 0)
	for detection in Tables.namedtupleiterator(detections)
		template = templates[detection.template, :]
		x_guess = [template.north, template.east, template.up]
		t_guess = samplestotimes * (detection.peak_sample + samples_pre)
		candidate = locate(detection.arrivals, x_guess, t_guess, template.weights)
	    if Optim.converged(candidate)
			north, east, up, delay = candidate.minimizer
        	residual = candidate.minimum
    	else
        	north, east, up = x_guess
        	delay = t_guess
        	residual = missing
    	end
		push!(norths, north)
		push!(easts, east)
		push!(ups, up)
		push!(times, detection.datastart + delay)
		push!(residuals, residual)
	end
	detections.north = norths
	detections.east = easts
	detections.up = ups
	detections.time = times
	detections.residual = residuals
	detections
end

# ╔═╡ 31ee4bf4-e1bc-461f-b3e5-d2a688644bda
histogram2d(detections.north, detections.east)

# ╔═╡ c1e952a4-83c4-4419-a0bc-bc579e643198
md"## Compute magnitudes and cross correlations"

# ╔═╡ 57eb3a05-55ca-4ef7-a780-b62acc327db7
let
	N = nrow(detections)
	correlation_mean = Vector{Float64}(undef, N)
	correlation_std = Vector{Float64}(undef, N)
	magnitudes = Vector{Float64}(undef, N)
	for (n, detection) in enumerate(Tables.namedtupleiterator(detections))
		template = templates[detection.template, :]
		ws = weights(template.weights)
		cc_mean, cc_std = mean_and_std(detection.correlations, ws)
		amps_squared = maximum.(map(series -> series.^2, detection.data))
		x0 = [detection.north, detection.east, detection.up]
		dists_squared = map(xs -> dot(xs - x0, xs - x0), sensors)
		mag = 0.5log10(mean(amps_squared .* dists_squared, ws)) - 2
		correlation_mean[n] = cc_mean
		correlation_std[n] = cc_std
		magnitudes[n] = mag
	end
	detections.magnitude = magnitudes
	detections.correlation_mean = correlation_mean
	detections.correlation_std = correlation_std
	detections
end

# ╔═╡ 73942647-c38a-47ea-bdb1-3905ba3fc013
let
	plt = plot()
	histogram!(plt, detections.magnitude, label="detections", fillalpha=0.5)
	histogram!(plt, good_templates.magnitude, label="templates", fillalpha=0.5)
	histogram!(plt, good_templates.magnitude_recomputed, label="templates (recomputed)", fillalpha=0.5)
	plt
end

# ╔═╡ c0cca883-4b3c-43e5-bccf-ebb09037d7a0
summarystats(detections.magnitude)

# ╔═╡ ebc8816e-c63b-4e7e-97ed-c7e62d3e51e5
summarystats(good_templates.magnitude)

# ╔═╡ 429b10fa-9de1-463f-a3bc-84d22208f896
summarystats(good_templates.magnitude_recomputed)

# ╔═╡ b0d7873f-d915-45ac-8929-0b2d1ef159d6
md"## Saving the catalogue"

# ╔═╡ dfbde837-238b-4184-b72d-10cde675d132
md"""Experiment starts at $(@bind starttimestr TextField(default="2022-04-07_18-18-02"))"""

# ╔═╡ 2bb338c9-2074-4290-a620-1f0ce9ebf154
starttime = DateTime(starttimestr, "yyyy-mm-dd_HH-MM-SS")

# ╔═╡ 80413628-e850-4423-a3c8-dc9a1da75821
begin
	secs = floor.(Int, 1e-6detections.time)
	usecs = detections.time - 1e6secs

	dates = starttime .+ Second.(secs)
	
	detections.year = year.(dates)
	detections.month = month.(dates)
	detections.day = day.(dates)
	detections.hour = hour.(dates)
	detections.minute = minute.(dates)
	detections.second = second.(dates)
	detections.microsecond = usecs
	detections
end

# ╔═╡ 2ed676c7-7346-40cc-b79e-301e3f8662cb
csv_columns = [:year, :month, :day, :hour, :minute, :second, :microsecond, :template, :north, :east, :up, :magnitude, :correlation_mean, :correlation_std, :residual]

# ╔═╡ b18c25c7-0dd3-4274-9ecb-77e52e936432
let df = copy(detections), cols = copy(csv_columns)
	for n = 1:16
		push!(cols, Symbol("cc$n"))
		df[!, "cc$n"] = map(xs -> xs[n], df.correlations)
	end
	CSV.write("catalogue.csv", df[!, cols])
end

# ╔═╡ 464769f3-57a5-4434-81e1-6edae4046a6d
CSV.write("good_templates.csv", good_templates[!, [:index, :magnitude, :magnitude_recomputed]])

# ╔═╡ Cell order:
# ╠═f00d14ea-b8db-4bf2-bfdf-90a4a3cb3ccf
# ╠═8b4426b8-588b-49ad-83c5-a79ed698d704
# ╠═3c07c33b-7f91-4e2d-a27a-09a8924f328c
# ╠═b59dd2c0-4c3e-4c3d-a4d4-f8da9dcbe286
# ╠═76e2e659-c708-4a08-8263-def6c3dfa930
# ╠═533d8e9e-40ff-4740-bbd9-8199cf0bcef6
# ╠═a7c96f5f-db00-48ac-8945-c268200fd868
# ╠═89c57605-3aea-4335-adae-55c40c972d26
# ╠═f07ab63f-dfb4-493c-a40c-6b3363a858c3
# ╠═356cc4c2-e565-4847-9286-cfc1f835654d
# ╠═3ec5251a-2e3e-4344-bb1f-fcd79daf9714
# ╠═79f1487a-0dc8-483b-a100-ecd176b7dd92
# ╟─57b35604-727b-4eac-bcbd-ea302e4c79a2
# ╠═2fc4fdfe-81b2-45f1-9ed4-b8ad5bb86ee3
# ╠═43f3ae0e-9eb0-47d1-b750-b04355f12082
# ╟─7987b27b-c7ae-4f97-93ee-b83971ec0248
# ╠═6d399562-aed4-42a6-b4ee-f1b6cb2c9a4e
# ╠═fea071ff-179b-46b0-8bff-953712eb8fd8
# ╠═fd864a98-14ee-4bdd-9a8d-52e80ee4f5a1
# ╟─6ea0c487-57e6-4ade-bd45-15226df212ec
# ╠═2c1941d0-ab86-40cb-8897-d1898e41f27c
# ╟─102e4534-6be4-4436-be36-69726a393253
# ╠═035a667d-e948-4dbe-8ae0-bcfd571a2da9
# ╠═20ac88b1-5a8a-4919-b6cd-8b4a7d829349
# ╠═12950cbd-cfe8-4a53-a93c-7da9aec408bc
# ╠═81154535-0576-4a69-b906-dabc3d7c8f9d
# ╠═d4b0ffe4-92f8-4a25-a259-aa9175f39933
# ╠═9a1c9fe2-8f1d-4f2a-b0b4-ef1bf91affdc
# ╠═aed075ff-6ca6-4dd6-bb64-ab75177f654e
# ╠═fe822a34-bdc7-486f-a225-2ed2c15a8371
# ╟─26c5d8c7-ce56-41d6-bab5-63a0f1cc369e
# ╠═417f915e-0013-48b6-bc22-2b04125d4c18
# ╠═b963608d-92a1-4416-9a34-bb60b63b6406
# ╠═9205dd4a-c67e-470c-9bd1-06fa03187b34
# ╠═caaab003-d9be-41fa-99cf-d9464db0b612
# ╠═443cefe2-f349-4cd4-be3e-6b83537ae6df
# ╠═3d6696cc-be33-4c5a-99f5-15fa1317a308
# ╠═74e83a60-31a6-42ae-8367-c573c34483d5
# ╠═31ee4bf4-e1bc-461f-b3e5-d2a688644bda
# ╟─c1e952a4-83c4-4419-a0bc-bc579e643198
# ╠═57eb3a05-55ca-4ef7-a780-b62acc327db7
# ╠═73942647-c38a-47ea-bdb1-3905ba3fc013
# ╠═c0cca883-4b3c-43e5-bccf-ebb09037d7a0
# ╠═ebc8816e-c63b-4e7e-97ed-c7e62d3e51e5
# ╠═429b10fa-9de1-463f-a3bc-84d22208f896
# ╟─b0d7873f-d915-45ac-8929-0b2d1ef159d6
# ╟─dfbde837-238b-4184-b72d-10cde675d132
# ╠═2bb338c9-2074-4290-a620-1f0ce9ebf154
# ╠═80413628-e850-4423-a3c8-dc9a1da75821
# ╠═2ed676c7-7346-40cc-b79e-301e3f8662cb
# ╠═b18c25c7-0dd3-4274-9ecb-77e52e936432
# ╠═464769f3-57a5-4434-81e1-6edae4046a6d
