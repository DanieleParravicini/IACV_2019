# pip install opencv-contrib-python

import cv2
import numpy as np
import time
#import function for normal distribution
from scipy.stats import norm
import math  
import threading
import matplotlib.pyplot as plt

verbose = False

class GaussianMixtureModelPixel:
  LARGE_VARIANCE  = 20
  MATCH_THRESHOLD = 2.5
  ALPHA           = 0.01
  LOW_WEIGHT      = 0.01
  N = 0
  T = 0.7
  EPSILON = 0.001
  
  # constructor   
  def __init__(self,n = 3):  
    self.averages     = []
    self.variances    = []
    self.weights      = []
    self.N            = n

  def renormalize_weights(self):
    weights_tot  = sum(self.weights)
    self.weights = list(map(lambda w: w/weights_tot, self.weights))

  def background_index(self):
    cumulative_w = 0
    b_index = 0
    while( cumulative_w < self.T and b_index < len(self.weights)):
      cumulative_w += self.weights[b_index]
      b_index+=1

    return b_index -1

  def update(self, new_x):
    new_x = int(new_x)
    i = self.match(new_x)

    if verbose : 
      print('ENTERING UPDATE:')
      print_stats()
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
      rho = self.ALPHA#*norm.pdf(new_x, self.averages [i], math.sqrt(self.variances[i]) )
      #rho  = self.ALPHA /max([ self.weights[i] , self.ALPHA])
      if verbose : 
        print('rho:',rho)
      self.averages [i] =  int(self.averages[i]   + rho   *  (new_x - self.averages[i]))
      new_variance = int(self.variances[i]  + rho   * ((new_x - self.averages[i])**2 - self.variances[i] ) )
      new_variance = max(new_variance, 1)
      assert(new_variance >= 1)
      self.variances[i] = new_variance

    #update weights
    self.weights = list(map(lambda j,w : w + self.ALPHA*( j==i - w), range(len(self.weights)), self.weights ))
    self.renormalize_weights()

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
    b_index = self.background_index()
    is_background = (i == b_index)

    if verbose:
      print('BEFORE LEAVING UPDATE:')
      print_stats()
      print('is background: ', is_background)

    return not is_background
  
  #the index of the distribution matched or -1
  def match(self, x):
    
    for i in range(len(self.variances)):
      if( abs(int(x)- self.averages[i]) <= self.MATCH_THRESHOLD*math.sqrt(self.variances[i]) ):
          return i
    return -1
  
  def print_stats(self):
    for i in range(len(self.averages)):
        print('gaussian #',i, 
        'w/var', self.weights[i]/(self.variances[i]+self.EPSILON), 
        ' avg: ', self.averages[i], 
        ' var: ', self.variances[i], 
        'weight', self.weights[i] )
      

def update_background(thread_id, num_thread, frame, state, output):
  
  tot_element = frame.shape[0]*frame.shape[1]
  num_element = math.ceil(tot_element/num_thread) 
  start = num_element*(thread_id-1) 
  end   = start+num_element
  for i in range(start,end):
    y = i//frame.shape[1]
    x = i % frame.shape[1]
    output[y,x] = state[y][x].update(frame[y,x,0])
  
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
  print(nr_frame, frame.shape)
  nr_frame +=1 
  if ret == True:
    # Display the resulting frame
    test = np.full(frame.shape[:2],0)

    #fill state for the first time
    if state is None:
      state = []
      for i in range(frame.shape[0]):
        state.append([])
        for j in range(frame.shape[1]):
          state[i].append(GaussianMixtureModelPixel(3))
    

    #update
    nr_threads = 32
    threads = []
    unshaped_frame = np.reshape(frame,-1)
    
    for i in range(nr_threads):
      x = threading.Thread(target=update_background, args=(i, nr_threads, frame, state, test))
      x.start()
      threads.append(x)
    
    for t in threads:
      t.join()

    '''for i in range(100,200):#range(frame.shape[0]//2):
      for j in range(100,200):#range(frame.shape[1]//2):
        if((i,j) not in stats ):
          stats[(i,j)] = []
        stats[(i,j)].append(frame[i,j,0])
        test[i,j] = state[i][j].update(frame[i,j,0])'''
    
    test = test.astype(np.uint8)  #convert to an unsigned byte
    test*=255
    #print an overlay
    result = np.concatenate([ np.expand_dims(np.zeros(frame.shape[:2]).astype(np.uint8), axis=2),
                              np.expand_dims(frame[:,:,0], axis=2),
                              np.expand_dims(test, axis=2)
                              
                              ], axis=2)
    #print(result.shape)
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
