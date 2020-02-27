# pip install opencv-contrib-python

import cv2
import numpy as np
import time
#import function for normal distribution
from scipy.stats import norm
import math  
import matplotlib.pyplot as plt

verbose = False

class GaussianMixtureModelPixel:
  LARGE_VARIANCE  = 20
  MATCH_THRESHOLD = 2.5
  ALPHA           = 0.01
  LOW_WEIGHT      = 0.001
  N = 0
  T = 0.7
  EPSILON = 0.001
  

  variances = []
  averages  = []
  weights   = []
  
  # constructor   
  def __init__(self,n = 3):  
    self.averages     = []
    self.variances    = []
    self.weights      = []
    self.N            = n

  def update(self, new_x):
    if verbose : 
      print('ENTERING UPDATE:')
      for i in range(len(self.averages)):
        print('gaussian #',i, 
        'w/var', self.weights[i]/(self.variances[i]+self.EPSILON), 
        ' avg: ', self.averages[i], 
        ' var: ', self.variances[i], 
        'weight', self.weights[i] )

    i = self.match(new_x)

    if verbose : 
      print('gaussian matched by',new_x, ':', i)
    
    #is match found?
    if( i == -1):
      #if no match is found
      #we have to create a new Gaussian
      #have we reached N gaussian?
      if( len(self.averages) >= self.N):
        #if yes just replace the values of the last distribution
        i = self.N-1
      else:
        #if no add a new distribution
        #assign i to the new distribution
        i = len(self.averages)
        self.weights.append(0)
        self.averages.append(0)
        self.variances.append(0)
      #update values
      self.weights  [i] = 0
      self.averages [i] = new_x
      self.variances[i] = self.LARGE_VARIANCE
    else :
      #if a match is found we have to modify the current values of the distributions
      #rho = self.ALPHA*norm.pdf(new_x, self.averages [i], math.sqrt(self.variances[i]) )
      rho = self.ALPHA 
      if verbose : 
        print('rho:',rho)
      self.averages [i] = self.averages[i]   + rho   * (new_x - self.averages[i])
      self.variances[i] = (self.variances[i] + rho   * ((new_x - self.averages[i])**2 - self.variances[i] ) )
    #update weights
    for j in range(len(self.weights)):
      self.weights  [j] = (1 - self.ALPHA) * self.weights[j] + self.ALPHA * (j==i)
    #renormalize weights
    sum_weights  = sum(self.weights)
    self.weights = list(map(lambda x: x/ sum_weights, self.weights))
    #reorder according to self.weights[.]/self.variance[.]
    
    weight     = self.weights   [i]
    average    = self.averages  [i]
    variance   = self.variances [i]
    del self.weights   [i]
    del self.averages  [i]
    del self.variances [i]
    #weights of gaussian with lower index have at most diminuished hence 
    #no reason to test against them. start from old position of the distribution 
    j = i-1
    while( j >= 0 and self.weights[j]/(self.variances[j]+self.EPSILON) < weight/(variance+self.EPSILON) ):
      j = j-1
      
    j = j+1
    self.weights.insert(j, weight)
    self.averages.insert(j, average)
    self.variances.insert(j, variance)

    # the given pixel belongs to background if it
    # falls over T cumulative percentage in the 
    # gaussian list.
    
    is_background = sum(self.weights[:i]) <= self.T
    if verbose:
      print('BEFORE LEAVING UPDATE:')
      for i in range(len(self.averages)):
        print('gaussian #',i, 
        'w/var', self.weights[i]/(self.variances[i]+self.EPSILON), 
        ' avg: ', self.averages[i], 
        ' var: ', self.variances[i], 
        'weight', self.weights[i] )
    
    return not is_background
  
  #the index of the distribution matched or -1
  def match(self, x):
    
    for i in range(len(self.variances)):
      
      if( self.averages[i] - self.MATCH_THRESHOLD*math.sqrt(self.variances[i]) <  x   and
          self.averages[i] + self.MATCH_THRESHOLD*math.sqrt(self.variances[i]) >= x  ):
          return i
    return -1

state = None
stats = {}
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('CCTV.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
nr_frame = 0
while(cap.isOpened()):
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  #
  print('frame:' , nr_frame , 'frame dimension ', frame.shape)
  nr_frame +=1 
  if ret == True:
    # Display the resulting frame
    
    test = np.full(frame.shape[:2],0)

    #fill state for the first time
    if state is None:
      state = np.full(frame.shape[:2],GaussianMixtureModelPixel(4))
    #update
    for i in range(100,200):#range(frame.shape[0]//2):
      for j in range(100,200):#range(frame.shape[1]//2):
        if((i,j) not in stats ):
          stats[(i,j)] = []
        stats[(i,j)].append(frame[i,j,0])
        test[i,j] = state[i,j].update(frame[i,j,0])
    
    test = test.astype(np.uint8)  #convert to an unsigned byte
    test*=255
    #print an overlay
    result = np.concatenate([ np.expand_dims(np.zeros(frame.shape[:2]).astype(np.uint8), axis=2),
                              np.expand_dims(frame[:,:,0], axis=2),
                              np.expand_dims(test, axis=2)
                              
                              ], axis=2)
    print(result.shape)
    cv2.imshow('result',result)
    if(verbose):
      time.sleep(1)
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      '''for k in stats:
        print(k)
        plt.figure()
        plt.plot(stats[k])
        plt.show()
        time.sleep(100)
      '''
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
