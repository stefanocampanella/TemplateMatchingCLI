### A Pluto.jl notebook ###
# v0.19.11

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
md"""Path of the catalogue JLD $(@bind cataloguepath TextField(default="../data/detections.jld2"))"""

# ╔═╡ 43f3ae0e-9eb0-47d1-b750-b04355f12082
begin
	detections = load(cataloguepath, "detections")
	# filter false positives
	filter!(:correlations => ccs -> count(>=(0.5), ccs) > 8, detections)
	# filter faulty template
	filter!(:template => !=(2263), detections)
end

# ╔═╡ 7987b27b-c7ae-4f97-93ee-b83971ec0248
md"## Read templates catalogue"

# ╔═╡ 6d399562-aed4-42a6-b4ee-f1b6cb2c9a4e
md"""Path of the templates JLD $(@bind templatespath TextField(default="../data/catalogue.jld2"))"""

# ╔═╡ fea071ff-179b-46b0-8bff-953712eb8fd8
templates = load(templatespath, "catalogue")

# ╔═╡ fd864a98-14ee-4bdd-9a8d-52e80ee4f5a1
let ts = dropmissing(templates)
	sort([n for n in 1:5254 if n ∉ Set(ts.index)])
end

# ╔═╡ 6ea0c487-57e6-4ade-bd45-15226df212ec
md"## Read sensors positions"

# ╔═╡ 2c1941d0-ab86-40cb-8897-d1898e41f27c
sensors = load(templatespath, "sensorscoordinates")

# ╔═╡ c1e952a4-83c4-4419-a0bc-bc579e643198
md"## Compute magnitudes and cross correlations"

# ╔═╡ 57eb3a05-55ca-4ef7-a780-b62acc327db7
let
	N = nrow(detections)
	correlation_mean = Vector{Float64}(undef, N)
	correlation_std = Vector{Float64}(undef, N)
	magnitude_mean = Vector{Float64}(undef, N)
	magnitude_std = Vector{Float64}(undef, N)
	for (n, detection) in enumerate(Tables.namedtupleiterator(detections))
		template = templates[detection.template, :]
		ws = weights(template.weights)
		rel_mags = relative_magnitude.(template.data, detection.data)
		rmag_mean, rmag_std = mean_and_std(rel_mags, ws)
		cc_mean, cc_std = mean_and_std(detection.correlations, ws)
		magnitude_mean[n] = template.magnitude + rmag_mean
		magnitude_std[n] = rmag_std
		correlation_mean[n] = cc_mean
		correlation_std[n] = cc_std
	end
	detections.magnitude_mean = magnitude_mean
	detections.magnitude_std = magnitude_std
	detections.correlation_mean = correlation_mean
	detections.correlation_std = correlation_std
	detections
end

# ╔═╡ 26c5d8c7-ce56-41d6-bab5-63a0f1cc369e
md"## Localize detections"

# ╔═╡ 3d6696cc-be33-4c5a-99f5-15fa1317a308
line_element_squared(x, t) = (dot(x, x) - 0.67^2 * t^2)^2

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
		t_guess = detection.peak_sample + 50
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

# ╔═╡ 102e4534-6be4-4436-be36-69726a393253
md" ## Statistics of the detections "

# ╔═╡ 035a667d-e948-4dbe-8ae0-bcfd571a2da9
begin
	template_counts = combine(groupby(detections, :template), nrow => :count)
	sort!(template_counts, :count, rev=true)
end

# ╔═╡ 20ac88b1-5a8a-4919-b6cd-8b4a7d829349
bar(template_counts.template, template_counts.count, label=nothing)

# ╔═╡ 1efe262f-ec75-4012-bb2c-7d15c348790d
histogram(template_counts.count)

# ╔═╡ 73942647-c38a-47ea-bdb1-3905ba3fc013
histogram(templates.magnitude, label=nothing)

# ╔═╡ 05c28a49-eddd-4a9d-bd5f-ed130f636f67
histogram(detections.magnitude_mean, label=nothing)

# ╔═╡ 80413628-e850-4423-a3c8-dc9a1da75821
begin
	secs = floor.(Int, 1e-6detections.time)
	usecs = detections.time - 1e6secs

	dates = DateTime(2021, 1, 12, 20, 24, 18) .+ Second.(secs)
	
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
csv_columns = [:year, :month, :day, :hour, :minute, :second, :microsecond, :template, :north, :east, :up, :magnitude_mean, :magnitude_std, :correlation_mean, :correlation_std, :residual]

# ╔═╡ b18c25c7-0dd3-4274-9ecb-77e52e936432
let df = copy(detections), cols = copy(csv_columns)
	for n = 1:16
		push!(cols, Symbol("cc$n"))
		df[!, "cc$n"] = map(xs -> xs[n], df.correlations)
	end
	CSV.write("catalogue.csv", df[!, cols])
end

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
# ╟─57b35604-727b-4eac-bcbd-ea302e4c79a2
# ╠═2fc4fdfe-81b2-45f1-9ed4-b8ad5bb86ee3
# ╠═43f3ae0e-9eb0-47d1-b750-b04355f12082
# ╟─7987b27b-c7ae-4f97-93ee-b83971ec0248
# ╠═6d399562-aed4-42a6-b4ee-f1b6cb2c9a4e
# ╠═fea071ff-179b-46b0-8bff-953712eb8fd8
# ╠═fd864a98-14ee-4bdd-9a8d-52e80ee4f5a1
# ╟─6ea0c487-57e6-4ade-bd45-15226df212ec
# ╠═2c1941d0-ab86-40cb-8897-d1898e41f27c
# ╟─c1e952a4-83c4-4419-a0bc-bc579e643198
# ╠═57eb3a05-55ca-4ef7-a780-b62acc327db7
# ╟─26c5d8c7-ce56-41d6-bab5-63a0f1cc369e
# ╠═caaab003-d9be-41fa-99cf-d9464db0b612
# ╠═443cefe2-f349-4cd4-be3e-6b83537ae6df
# ╠═3d6696cc-be33-4c5a-99f5-15fa1317a308
# ╠═74e83a60-31a6-42ae-8367-c573c34483d5
# ╟─102e4534-6be4-4436-be36-69726a393253
# ╠═035a667d-e948-4dbe-8ae0-bcfd571a2da9
# ╠═20ac88b1-5a8a-4919-b6cd-8b4a7d829349
# ╠═1efe262f-ec75-4012-bb2c-7d15c348790d
# ╠═73942647-c38a-47ea-bdb1-3905ba3fc013
# ╠═05c28a49-eddd-4a9d-bd5f-ed130f636f67
# ╠═79f1487a-0dc8-483b-a100-ecd176b7dd92
# ╠═80413628-e850-4423-a3c8-dc9a1da75821
# ╠═2ed676c7-7346-40cc-b79e-301e3f8662cb
# ╠═b18c25c7-0dd3-4274-9ecb-77e52e936432
