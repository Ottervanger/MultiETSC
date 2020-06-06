#=== Deal with inputs.
if ARGV.length < 5
	puts "saps_wrapper.rb is a wrapper for the SAPS algorithm."
	puts "Usage: ruby saps_wrapper.rb <instance_relname> <instance_specifics> <cutoff_time> <cutoff_length> <seed> <params to be passed on>."
	exit -1
end
cnf_filename = ARGV[0]
instance_specifics = ARGV[1]
cutoff_time = ARGV[2].to_f
cutoff_length = ARGV[3].to_i
seed = ARGV[4].to_i

#=== Here I assume instance_specifics only contains the desired target quality or nothing at all for the instance, but it could contain more (to be specified in the instance_file or instance_seed_file)
if instance_specifics == ""
	qual = 0
else
	qual = instance_specifics.split[0]
end

paramstring = ARGV[5...ARGV.length].join(" ")

#=== Build algorithm command and execute it.
cmd = "example_saps/ubcsat.exe -alg saps #{paramstring} -inst #{cnf_filename} -cutoff #{cutoff_length} -timeout #{cutoff_time} -target #{qual} -seed #{seed} -r stats stdout default,best"

filename = "example_saps/ubcsat_output#{rand}.txt"
exec_cmd = "#{cmd} > #{filename}"

puts "Calling: #{exec_cmd}"
system exec_cmd

#=== Parse algorithm output to extract relevant information for ParamILS.
solved = nil
runtime = nil
runlength = nil
best_sol = nil

File.open(filename){|file|
	while line = file.gets
		if line =~ /SuccessfulRuns = (\d+)/
			numsolved = $1.to_i
			if numsolved > 0
				solved = "SAT"
			else
				solved = "TIMEOUT"
			end
		end
		if line =~ /CPUTime_Mean = (.*)$/
			runtime = $1.to_f
		end
		if line =~ /Steps_Mean = (\d+)/
			runlength = $1.to_i
		end
		if line =~ /BestSolution_Mean = (\d+)/
			best_sol = $1.to_i
		end
	end
}
File.delete(filename)
puts "Result for ParamILS: #{solved}, #{runtime}, #{runlength}, #{best_sol}, #{seed}"
