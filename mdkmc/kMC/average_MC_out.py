#!/usr/bin/python

import numpy as np
import argparse
import ipdb, sys

def avg(filename, lines, intervals, columns, time_columns, variance, verbose=False):

	data=np.loadtxt(filename, usecols=columns)
	time = np.loadtxt(filename, usecols=time_columns)

	if verbose == True:
		print "#Shape of data:", data.shape

	if intervals != None and intervals < data.shape[0]:
		data = data[:intervals * lines]
	else:
		intervals = data.shape[0]/lines

	if verbose == True:
		print "#Intervals:", intervals

	data = data.flatten().reshape(intervals, lines, len(columns))

	avg = data.mean(axis=0)

	if variance == True:
		var = data[:,:,:].var(axis=0)
		return time, avg, var
	else:
		return time[:avg.shape[0]], avg



def read_from_configfile(config_filename):
	data = dict()
	with open(config_filename, "r") as f:
		for line in f:
			if line.lstrip()[0] != "#":
				if len(line.split()) > 2:
					data[line.split()[0]] = line.split()[1:]
				else:
					data[line.split()[0]] = line.split()[1]
		#~ for line in data:
			#~ print line
	return data

def get_observable_names(outfilename):
	with open(outfilename, "r") as f:
		comments = [line for line in list(f) if line[0] == "#"]
	return comments[-2][1:].split()

def main(*args):

	parser=argparse.ArgumentParser(description="Average KMC output. Assuming time in first column")
	parser.add_argument("--file", help="KMC output")
	parser.add_argument("--lines", type=int, help="Specify length of averaged interval")
	parser.add_argument("--intervals", type=int, help="Average over how many intervals?")
	parser.add_argument("--columns", "-c", type=int, nargs="+", help="Over which columns to average (starting with zero)")
	parser.add_argument("--time_column", "-t", type=int, help="Specify time column")
	parser.add_argument("--variance", action="store_true", help="Also output variance")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
	parser.add_argument("--config_load", help="Config file of KMC run")
	args = parser.parse_args()
	
	if args.config_load != None:
		data = read_from_configfile(args.config_load)
		try:
			kmc_out = data["output"]
		except KeyError:
			print "#No kmc outputfile specified in kmc config"
			if args.file == None:
				print "#Neither is --file specified"
				print "#Exiting"
				sys.exit(1)
			else:
				print "#Using --file {}".format(args.file)
				kmc_out = args.file
			#~ ipdb.set_trace()
		total_lines = int(data["sweeps"])/int(data["print_freq"])
		intervals = int(data["sweeps"])/int(data["reset_freq"])
		lines = total_lines / intervals
		time_columns = 0, 1
		columns = [2, 3, 4, 5, 6]
		
		result = avg(kmc_out, lines, intervals, columns, time_columns, args.variance, args.verbose)

	else:
		kmc_out = args.file
		result = avg(kmc_out, args.lines, args.intervals, args.columns, args.time_column, args.variance, args.verbose)
	
	comments = get_observable_names(kmc_out)
	#~ print comments[0], comments[1],
	#~ for column in columns:
		#~ print comments[column],
	#~ print""
	
	total_columns = len(time_columns) + len(columns)
	if args.variance == True:
		print "#{:12} {:12}"+6*" {:16}"+4*" {:8}".format(comments[0], comments[1], comments[2], comments[2]+"_var",  comments[3], comments[3]+"_var", comments[4], comments[4]+"_var", comments[5], comments[5]+"_var", comments[6], comments[6]+"_var")
		format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msdvar[0]:12.4f} {msd[1]:12.4f} {msdvar[1]:12.4f} {msd[2]:12.4f} {msdvar[2]:12.4f} {autocorr:6.2f} {autocorrvar:6.2f} {jumps:6.2f} {jumpsvar:6.2f}"
		t, a, v = result
	
		for i in xrange(a.shape[0]):
			print format_string.format(t=t, msd=a[i,0:3], msdvar=v[i,0:3], autocorr=a[i,3], autocorrvar=v[i,3], jumps=a[i,4], jumpsvar=v[i,4])
			#~ print t[i,0], t[i,1],
			#~ for j in xrange(a.shape[1]):
				#~ print a[i,j], v[i,j],
			#~ print ""
		
	else:
		print "#", " ".join(["{:<12}", "{:<12}", 3*"{:<16}", 2*"{:<8}"]).format(*comments)
		format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msd[1]:12.4f} {msd[2]:12.4f} {autocorr:6.2f} {jumps:6.2f}"
		t, a = result
		#~ ipdb.set_trace()
		for i in xrange(a.shape[0]):
			print format_string.format(t=t[i], msd=a[i,:3], autocorr=a[i,3], jumps=a[i,4])
			#~ print t[i,0], t[i,1],
			#~ for j in xrange(a.shape[1]):
				#~ print a[i,j],
			#~ print ""

if __name__ == "__main__":
	main()


