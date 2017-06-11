import numpy as np

def detect_cusum(data, basesd, dt, threshhold = 10, stepsize = 3, 
                 minlength = 1000, maxstates = -1):

#    dt = 1
#    threshhold = 1
#    basesd = np.std(data[0:100000])
#    print 'basesd = ' + string(basesd)
#    stepsize = 3
#    minlength = 1000

    logp = 0 #instantaneous log-likelihood for positive jumps
    logn = 0 #instantaneous log-likelihood for negative jumps
    cpos = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for positive jumps
    cneg = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for negative jumps
    gpos = np.zeros(len(data), dtype='float64') #decision function for positive jumps
    gneg = np.zeros(len(data), dtype='float64') #decision function for negative jumps
    edges = np.array([0], dtype='int64') #initialize an array with the position of the first subevent - the start of the event
    anchor = 0 #the last detected change
    length = len(data)
    mean = data[0]
    variance = basesd**2
    k = 0
    nStates = 0
    varM = data[0]
    varS = 0
    mean = data[0]
    sThreshhold = threshhold
    sStepSize = stepsize

    while k < length-1:
            k += 1
            varOldM = varM #algorithm to calculate running variance, details here: http://www.johndcook.com/blog/standard_deviation/
            varM = varM + (data[k] - varM)/float(k+1-anchor)
            varS = varS + (data[k] - varOldM) * (data[k] - varM)
            variance = varS / float(k+1-anchor)
            mean = ((k-anchor) * mean + data[k])/float(k+1-anchor)
            if (variance == 0):                 # with low-precision data sets it is possible that two adjacent values are equal, in which case there is zero variance for the two-vector of sample if this occurs next to a detected jump. This is very, very rare, but it does happen.
                    variance = basesd*basesd # in that case, we default to the local baseline variance, which is a good an estimate as any.
                    print('entered')
            logp = stepsize*basesd/variance * (data[k] - mean - stepsize*basesd/2.) #instantaneous log-likelihood for current sample assuming local baseline has jumped in the positive direction
            logn = -stepsize*basesd/variance * (data[k] - mean + stepsize*basesd/2.) #instantaneous log-likelihood for current sample assuming local baseline has jumped in the negative direction
            cpos[k] = cpos[k-1] + logp #accumulate positive log-likelihoods
            cneg[k] = cneg[k-1] + logn #accumulate negative log-likelihoods
            gpos[k] = max(gpos[k-1] + logp, 0) #accumulate or reset positive decision function
            gneg[k] = max(gneg[k-1] + logn, 0) #accumulate or reset negative decision function
            if (gpos[k] > threshhold or gneg[k] > threshhold):
                    if (gpos[k] > threshhold): #significant positive jump detected
                            jump = anchor + np.argmin(cpos[anchor:k+1]) #find the location of the start of the jump
                            if jump - edges[nStates] > minlength:
                                    edges = np.append(edges, jump)
                                    nStates += 1
                    if (gneg[k] > threshhold): #significant negative jump detected
                            jump = anchor + np.argmin(cneg[anchor:k+1])
                            if jump - edges[nStates] > minlength:
                                    edges = np.append(edges, jump)
                                    nStates += 1
                    anchor = k
                    cpos[0:len(cpos)] = 0 #reset all decision arrays
                    cneg[0:len(cneg)] = 0
                    gpos[0:len(gpos)] = 0
                    gneg[0:len(gneg)] = 0
                    mean = data[anchor]
                    varM = data[anchor]
                    varS = 0
            if maxstates > 0:
                if nStates > maxstates:
                    print('too sensitive')
                    print(threshhold,stepsize)
                    nStates = 0
                    k = 0
                    stepsize = stepsize*1.1
                    threshhold = threshhold*1.1
                    logp = 0 #instantaneous log-likelihood for positive jumps
                    logn = 0 #instantaneous log-likelihood for negative jumps
                    cpos = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for positive jumps
                    cneg = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for negative jumps
                    gpos = np.zeros(len(data), dtype='float64') #decision function for positive jumps
                    gneg = np.zeros(len(data), dtype='float64') #decision function for negative jumps
                    edges = np.array([0], dtype='int64') #initialize an array with the position of the first subevent - the start of the event
                    anchor = 0 #the last detected change
                    length = len(data)
                    mean = data[0]
                    variance = basesd**2
                    k = 0
                    nStates = 0
                    varM = data[0]
                    varS = 0
                    mean = data[0]
                    
    edges = np.append(edges, len(data)) #mark the end of the event as an edge
    nStates += 1


    cusum = dict()
    cusum['CurrentLevels'] = [np.average(data[int(edges[i]+minlength):int(edges[i+1])]) for i in range(nStates)] #detect current levels during detected sub-events
    cusum['EventDelay'] = edges * dt #locations of sub-events in the data
    cusum['Threshold'] = threshhold #record the threshold used
    cusum['stepsize'] = stepsize
    cusum['jumps'] = np.diff(cusum['CurrentLevels'])
    #self.__recordevent(cusum)

    return cusum
