import numpy as np
import scipy.odr
import math


class Box:
  def __init__(self,x0,y0,x1,y1,score,intIn=False):
    self._x0, self._y0 = x0, y0
    self._x1, self._y1 = x1, y1
    self._score = score
  # recover coords with min/max values
  def xMin(self):
    return min([self._x0,self._x1])
  def yMin(self):
    return min([self._y0,self._y1])
  def xMax(self):
    return max([self._x0,self._x1])
  def yMax(self):
    return max([self._y0,self._y1])
  def score(self): return self._score
  # derived variables
  def midpoint(self):
    xMid = (self._x0 + self._x1) / 2
    yMid = (self._y0 + self._y1) / 2
    if self._intIn: return int(xMid),(yMid)
    else: return xMid,yMid



# generated based on the scipy.ord doc page:
# https://docs.scipy.org/doc/scipy/reference/odr.html
def getOdrBeta(xL,yL):
  """The slope of the Demming regression line"""
  x,y = np.array(xL),np.array(yL)
  beta0 = np.polyfit(x,y,1)
  # the func to fit against
  f = lambda B,x: B[0]*x + B[1]
  linear = scipy.odr.Model(f)
  data = scipy.odr.RealData(x,y)
  odr = scipy.odr.ODR(data, linear, beta0=beta0)
  return odr.run().beta[0]

def getPosOnAxis(axSlope,axX,axY,dataX,dataY):
  """the slope "axSlope" & point "axX,axY" define the axis;
  finds the coordinate of an intersection point between the
  specified line and another line, perpendicular to the first,
  that also passes through the point "dataX,dataY".
  """
  newY = axY + (axSlope**2 * dataY) - axSlope*(axX - dataX)
  newY /= axSlope**2 + 1
  newX = (axSlope*axX + newY - axY) / axSlope
  return newX,newY

def getDist(xA,yA,xB,yB):
  return math.sqrt( (xA-xB)**2 + (yA-yB)**2 )

def getTargPoint(x,y,angle,dist):
  """determines a point at distance "dist" from
  point "x,y" in direction "angle, where
  zero angle == straight down
  """
  xT = x + (math.sin(angle) * dist)
  yT = y + (math.cos(angle) * dist)
  return xT,yT

def minAngDiff(angA,angB):
  """Finds the absolute value for the incident angle
  between angles A & B: smallest possible value.
  """
  aDiff = angA - angB
  if abs(aDiff) > math.radians(180):
    if aDiff > 0: return math.radians(360) - aDiff
    else: return math.radians(360) + aDiff
  else: return abs(aDiff)

def minAngNum(ang):
  """determines the smallest positive value
  for an angle (removes superfluous 360deg spins)
  """
  while ang < 0: ang += math.radians(360)
  ang %= math.radians(360)
  return ang

def getQuad(ang):
  """determines which quadrant an angle is in (1,2,3, or 4).
  REQUIRES minAngNum"""
  if ang < math.radians(180):
    if ang < math.radians(90): return 1
    else: return 2
  elif ang < math.radians(270): return 3
  else: return 4

def angMean(angA,angB,recN=0):
  """Determines the "average" between two angles.
  Defines the average as the halfway angle between
  the two angles across the shortest path around the circle.
  """
  angA,angB = minAngNum(angA),minAngNum(angB)
  qA,qB = getQuad(angA),getQuad(angB)
  abDiff = abs(qA-qB)
  if abDiff < 2: return np.mean([angA,angB])
  elif abDiff==2:
    if abs(angB-angA)==math.radians(180):
      # arbitrary direction choice, so just return the mean
      return np.mean([angA,angB])
    elif recN > 8:
      # must be ~= 180 since it went all the way around
      return np.mean([angA,angB])
    else:
      # rotate by 45 degrees to get into acceptable quadrants
      r45 = math.radians(45)
      return minAngNum(angMean(angA+r45,angB+r45,recN+1)-r45)
  else: return np.mean([angA,angB]) - math.radians(180)

class Ray:
  """A point and a direction (angle).  Implemented as a class
  so that I can bundle ray-related math functions into it.
  """
  def __init__(self,x,y,A):
    self._x = x
    self._y = y
    self._A = A
  def x(self): return self._x  
  def y(self): return self._y
  def A(self): return self._A
  def move(self,dist):
    return getTargPoint(self._x,self._y,self._A,dist)
  def dist(self,x,y):
    return getDist(x,y,self._x,self._y)
  def intersects(self,r):
    return self._A != r.A()
  def intersect(self,r):
    # no intersect if lines are parallel or equal
    if not(self.intersects(r)): return float('nan'),float('nan')
    # figured this out myself using the equation from 
    # "getTargPoint" above: solves for the distance
    # along THIS to move to get to the intersection
    numer = math.cos(r.A())*(self._x - r.x()) - math.sin(r.A())*(self._y - r.y())
    denom = math.sin(self._A)*math.cos(r.A()) - math.sin(r.A())*math.cos(self._A)
    return self.move(numer/denom)
  def opposite(self):
    newA = (self._A + math.radians(180)) % math.radians(360)
    return Ray(self._x,self._y,newA)

def getRay(xA,yA,xB,yB):
  """RETURNS a ray where point A is the origin and it points to B"""
  horz,vert = xB - xA, yB - yA
  # this order makes it degrees from vertical
  return Ray(xA,yA,math.atan2(horz,vert))
    

def sortBoxesOnAxis(bL):
  """defines an axis using ODR of the box centers,
  then sorts the boxes along that axis.
  """
  # sorts bottom-to-top
  if len(bL)==0: return []
  elif len(bL)==1: return [bL[0]]
  else:
    # sort the boxes by their position on the axis
    xyL = list(map(lambda b: b.midpoint(), bL))
    xL,yL = list(map(lambda i:i[0],xyL)),list(map(lambda i:i[1],xyL))
    # get the center point
    xMid,yMid = np.mean(xL),np.mean(yL)
    m = getOdrBeta(xL,yL)
    bsL = []
    for	bn in range(len(bL)):
      b = bL[bn]
      bX,bY = b.midpoint()
      bsL.append( (getPosOnAxis(m,xMid,yMid,bX,bY)[1],bn) )
    bsL.sort()
    bL = list(map(lambda i: bL[i[1]], bsL))
    bL.reverse()
    return bL


class Polygon:
  """just a series of points with the assumption
  that the last point connects back to the first.
  List elements should be in the order in which they 
  would be drawn;
  assumes that last point connects to first point;
  assumes legit geometry.
  """
  def __init__(self,xL,yL):
    self._xL = list(map(lambda i: i, xL))
    self._yL = list(map(lambda i: i, yL))
  def numSides(self): return len(self._xL)
  # sides are zero-indexed
  def sideLen(self,n):
    if n == len(self._xL): n = 0
    if n < len(self._xL) - 1:
      return getDist(self._xL[n],self._yL[n],self._xL[n+1],self._yL[n+1])
    else: return getDist(self._xL[-1],self._yL[-1],self._xL[0],self._yL[0])
  def corner(self,n):
    if n == len(self._xL): n = 0
    return self._xL[n],self._yL[n]



