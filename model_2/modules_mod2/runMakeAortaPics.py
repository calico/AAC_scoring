'''
This script/module uses spine-annotation models to identify the aortic
region of a DEXA scan and export it as a separate image.  ASSUMES that
the spine is on the left, aorta is on the right.
'''

#!/home/graham/anaconda2/bin/python
import os, cv2, sys
import argparse
import os
import numpy as np, math

import ml_tools as ML
import geometry_tools as GM
import file_tools as FT

        

# REQUIRES: vertEnList orders vertebrae from top to bottom;
#           if 'L4' is in the list, it must be 0
#           sample list: [('L3',1),('L4',0)]
class ImgGenerator:
  """This class manages the reading, analysis, and writing of files.
  It allows the images in a directory to be incremented through
  non-redundantly: with a single call to "makeImage", one image
  is read, analyzed, and a sub-image that should be zoomed in
  on the aorta is written out.
  """
  def __init__(self, imgDir, modSet, outDir, vertEnList, resizeOpt):
    """
    imgDir: the directory of images to be processed.  the contents
       will be read when the function is called, not again.
    modSet: a dictionary of the ML models needed for analysis. see
       'reqModL' below for a list of models.  Those with names ending
       in 'Det' are object-detectors (TfObjectDetector class); those with
       names ending in 'Class' are iamge classification models
       (TfClassApplyer class); those with names ending in 'Ok' are
       boolean models: these are wrappers for classification models
       that determine whether or not a specific class has been called
       (TfBooleanApplyaer class).
    outDir: the directory to which aortic images will be written.
    vertEnList: a specification of which vertebrae should be looked
       beside for finding the aortic image (a vestige of the development
       of this system, when I hadn't chosen this yet).
    resizeOpt: the iamge-resizing strategy to be applied. 'default' means
       that all images will be re-sized such that their height is 940 
       pixels; 'original' means as in Sethi et al, when applicable;
       'none' means no re-sizing.
    """
    # make sure all of the required models are there
    reqModL = ['vertDet', 'baseDet', 'vertOk', 'gapClass', 'fill1Det',
               'fillMuDet', 'baseOk', 'fillBotDet']
    for rm in reqModL:
      if not(modSet.has_key(rm)):
        raise ValueError("ImgGenerator needs this ML model for CalcBoxer: "+rm)
    self._imgMang = FT.ImageIterator(imgDir)
    self._modSet = modSet
    
    self._outDir = outDir
    self._elnL = map(lambda i:i, vertEnList)
    self._imgMang.initiateSort()

    # appropriate resizing will be achieved by pre-specifying a
    # resizing method that will be applied to all images
    if resizeOpt=='default':
      self._resizer = self._getDefaultResizer()
    elif resizeOpt=='original':
      self._resizer = self._getOriginalResizer()
    elif resizeOpt=='none':
      self._resizer = self._getNullResizer()
    else:
      raise ValueError('invalid resize option specified: '+resizeOpt)
    # stats on resizing for the output string
    self._nDefRsz,self._nNoRsz,self._n55Rsz = 0,0,0

  def hasImage(self): return self._imgMang.isSorting()

  # returns whether or not it wrote an image
  def makeImage(self):
    """Processes a single image.  Writes a file with a modified
    name (specifying which vertebrae were used) to the out dir.
    """
    didWrite = False
    # files & names
    finName = self._imgMang.getImgFile()
    foutNameL = os.path.basename(finName).split('.')
    foutNameL.append(foutNameL[-1])
    foutNameL[-2] = '_'.join(map(lambda i: i[0],self._elnL))
    foutName = '.'.join(foutNameL)
    foutName = os.path.join(self._outDir,foutName)
    # calculations on the image
    img = self._resizer(cv2.imread(finName), finName)
    calcBox = CalcBoxer(img,self._modSet)
    angD = calcBox.aortaRectAngles(self._elnL)
    rectD = calcBox.aortaRectangles(self._elnL)
    outImgL = []
    for el,other in self._elnL:
      if rectD.has_key(el):
        aortaImg = calcBox.rectangleImg(angD[el],rectD[el])
        outImgL.append(aortaImg)
    # only output if there are two images (all verts identified)
    if len(outImgL)==len(self._elnL):
      didWrite = True
      wdL = map(lambda i: i.shape[1], outImgL)
      htL = map(lambda i: i.shape[0], outImgL)
      outImgF = np.zeros((sum(htL),max(wdL),3),dtype=np.uint8)
      for vN in range(len(self._elnL)):
        htSt,htEnd = sum(htL[:vN]),sum(htL[:vN+1])
        outImgF[htSt:htEnd,0:wdL[vN],:] = outImgL[vN]
        if vN < len(self._elnL)-1:
          y = sum(htL[:vN+1])
          cv2.line(outImgF,(0,y),(max(wdL),y),(255,255,255),thickness=2)
      cv2.imwrite(foutName,outImgF)
    self._imgMang.moveToNext()
    return didWrite

  def resizeSummary(self):
    """Just returns a doc string.  Modifies nothing."""
    tL = []
    if self._nDefRsz > 0:
      tL.append(str(self._nDefRsz)+' images were re-sized uniformly to height=940.')
    if self._n55Rsz > 0:
      tL.append(str(self._n55Rsz)+' images were re-sized to 55% along each axis.')      
    if self._nNoRsz > 0:
      tL.append(str(self._nNoRsz)+' images were analyzed at their original sizes.')
    if len(tL)==0: return 'No images were analyzed.'
    else: return '\n'.join(tL)

  # the functions below provide methods for image
  # resizing that will also modify 'self' when called,
  # keeping track of how many times each re-sizing type
  # is applied (since 'original' will treat images differently
  # depending on their ID's).
  def _getNullResizer(self):
    def rsz(img,fname):
      self._nNoRsz += 1
      return img
    return rsz
  def _getDefaultResizer(self):
    def rsz(img,fname):
      self._nDefRsz += 1
      y,x = img.shape[:2]
      if x==0 or y==0: return img
      rsX = 940.0 / y
      return cv2.resize(img, (int(rsX*x),940))
    return rsz
  def _getOriginalResizer(self):
    f = open(permOldFileFile)
    lines = map(lambda i: i.rstrip(), f.readlines())
    f.close()
    idToSet = {}
    for i in lines: idToSet[i.split()[0]] = i.split()[1]
    rszNull = self._getNullResizer()
    rszDefault = self._getDefaultResizer()
    def resizeFunc(img,fname):
      fID = os.path.basename(fname).split('.')[0]
      if not(idToSet.has_key(fID)): return rszDefault(img,fname)
      elif idToSet[fID]=='A': return rszNull(img,fname)
      else:
        self._n55Rsz += 1
        y,x = img.shape[:2]
        if x==0 or y==0: return img
        x,y = x*.55,y*.55
        return cv2.resize(img, (int(x),int(y)))
    return resizeFunc


  
class CalcBoxer:
  """This class is the real workhorse for identification of the
  aortic region from an image and extracting it.  By itself, it
  represents a multi-step pipeline towards achieving its goal of
  returning an aortic sub-image from the input image.  Implementing
  it as a class made it easier for development: as steps are executed,
  additional data builds up, so storing that as class fields really
  sped up prototyping & development for me.  It is also useful for
  inspection apps that I've written, so I can easily store & extract
  the intermediate analysis features for printing/image-marking and
  reality-checking.

  This class is properly used by creating an instance for a DEXA
  image to-be-analyzed, then calling "aortaRectangles" and
  "aortaRectAngles" to get the boxes and their angles, respectively,
  adjacent to the requested vertebrae.  For each vertebra, calling
  "rectangleImg" with those two values will extract the specified 
  sub-image, now aligned with the vertical axis parallel to the spine.
  """
  def __init__(self,img,modSet,
               minScoreV=0.1,maxVertAngleDeg=45,minScorePost=0.5):
    """
    img: the full DEXA image to be analyzed (numpy array)
    modSet: a dictionary of the ML models needed for analysis. see
       'reqModL' below for a list of models.  Those with names ending
       in 'Det' are object-detectors (TfObjectDetector class); those with
       names ending in 'Class' are iamge classification models
       (TfClassApplyer class); those with names ending in 'Ok' are
       boolean models: these are wrappers for classification models
       that determine whether or not a specific class has been called
       (TfBooleanApplyaer class).
    """
    # make sure all of the required models are there
    reqModL = ['vertDet', 'baseDet', 'vertOk', 'gapClass', 'fill1Det',
               'fillMuDet', 'baseOk', 'fillBotDet']
    for rm in reqModL:
      if not(modSet.has_key(rm)):
        raise ValueError("CalcBoxer needs this ML model: "+rm)
    self._modSet = modSet
    self._minScV = minScoreV
    self._minScP = minScorePost
    self._maxAngV = math.radians(maxVertAngleDeg)
    # COPY the image
    if img is None: self._img = None
    else: self._img = np.dstack((img[:,:,0],img[:,:,1],img[:,:,2]))
    self._hasBx = False
    self._isRefined = False

  def refineSpineTrace(self):
    """The spine analysis pipeline.  It doesn't need to be
    called separately, any call that requires its results will
    call it if it wasn't already called.  Call it separately
    to prototype or visualize some component of the results
    it generates.
    """
    if not(self._img is None) and not(self._isRefined):
      if not(self._hasBx): self._makeAiBoxes()
      # filter by score
      self._boxScoreFilter(self._vertBL,self._vertExBL,self._minScV)
      # filter/add based on different properties
      self._applySpineAngleFilter()
      self._evaluateVerts()
      lastChanged = True
      for n in range(4):
        if lastChanged: lastChanged = self._addSkippedBoxes()
      self._isRefined = True
      # if L4 isn't found, look for it
      # CHANGED: look for L5
      if self._nForL4() < 1:
        self._addBottomBoxes()

  def aortaRectAngles(self,vertNameNL):
    """Returns a dictionary of the angles away from vertical that
    will be needed to generate an appropriately-rotated output
    image.  See "rectangleImg" to see how the angle will be used.
    Args:
    vertNameNL is a list of tupes: (name,pos-vs-L4), where up the
       spine adds positive values (so L3 is 1, L5 is -1)
    """
    if self._img is None: return {}
    nL4 = self._nForL4()
    if nL4==-1: return {}
    bL = GM.sortBoxesOnAxis(self._vertBL)
    if len(bL) < 2: return {}
    angPrL = map(lambda n: VertebraBoxPair(bL[n-1],bL[n]).angle(), 
                 range(1,len(bL)))
    angSgL = [ angPrL[0] ]
    for n in range(1,len(angPrL)):
      angSgL.append( GM.angMean(angPrL[n-1],angPrL[n]) )
    angSgL.append(angPrL[-1])
    retD = {}
    for vert,vN in vertNameNL:
      vN += nL4
      if vN >= 0 and len(angSgL) > vN:
        retD[vert] = angSgL[vN]
    return retD

  def aortaRectangles(self,vertNameNL):
    """Returns a polygon that is actually a rectangle, but may be
    at an odd, non-vertically-aligned angle.  It defines the region
    of the image that will be extracted by "rectangleImg".
    Args:
    vertNameNL is a list of tupes: (name,pos-vs-L4), where up the
       spine adds positive values (so L3 is 1, L5 is -1)
    """
    if self._img is None: return {}
    nL4 = self._nForL4()
    if nL4==-1: return {}
    bL = GM.sortBoxesOnAxis(self._vertBL)
    if len(bL) < 2: return {}
    # rays defining box edges 
    preRL,postRL = [],[]
    for n in range(1,len(bL)):
      xA,yA = bL[n-1].midpoint()
      vbp = VertebraBoxPair(bL[n-1],bL[n])
      xP,yP = vbp.midpoint()
      newR = GM.Ray(xP,yP,(vbp.angle()+math.radians(270)) % math.radians(360))
      if n == 1:
        halfD = GM.getDist(xA,yA,xP,yP)
        downR = GM.Ray(xA,yA,vbp.angle()).opposite()
        xD,yD = downR.move(halfD)
        preRL.append( GM.Ray(xD,yD,newR.A()) )
      preRL.append(newR)
      postRL.append(newR)
      if n+1 == len(bL):
        halfD = GM.getDist(xA,yA,xP,yP)
        upR = GM.Ray(xA,yA,vbp.angle())
        xU,yU = upR.move(halfD * 3)
        postRL.append( GM.Ray(xU,yU,newR.A()) )
    # rays defining box centers & upwards spine trajectories
    toboxRL,spineRL = [],[]
    for n in range(len(bL)):
      xA,yA = bL[n].midpoint()
      toboxRL.append( GM.Ray(xA,yA, GM.angMean(preRL[n].A(),postRL[n].A())) )
      spineRL.append( GM.Ray(xA,yA, (toboxRL[n].A()+math.radians(90)) % math.radians(360)) )
    # define the width of the spinal column across the lumbar vers
    lumbarNs = range(nL4,min([nL4+4,len(bL)]))
    if len(lumbarNs)==0: return {}
    widL = map(lambda b: b.xMax() - b.xMin(), bL)
    lumbarWids = map(lambda n: widL[n] * math.cos(GM.minAngDiff(spineRL[n].A(),math.radians(180))), lumbarNs)
    lumbarWids = filter(lambda i: i > 0, lumbarWids)  
    if len(lumbarWids)==0: return {}
    meanW_1t4 = np.mean(lumbarWids)
    walkOut = meanW_1t4 * 0.5
    width = meanW_1t4 * 0.75
    mbdist = width*0.5 + walkOut
    # rays defining box mid-points (pointing up)
    bmdRL = []
    for n in range(len(bL)):
      xM,yM = toboxRL[n].move(mbdist)
      bmdRL.append( GM.Ray(xM,yM, spineRL[n].A()) )
    # now I have everything I need to draw the boxes
    retD = {}
    for vert,vN in vertNameNL:
      vN += nL4
      if vN >= 0 and len(bL) > vN:
        # get the box top & bottom mid-points
        if bmdRL[vN].intersects(preRL[vN]) and bmdRL[vN].intersects(postRL[vN]):
          xT,yT = bmdRL[vN].intersect(preRL[vN])
          xB,yB = bmdRL[vN].intersect(postRL[vN])
          # get the top front & back points
          xC1,yC1 = GM.getTargPoint(xT,yT,toboxRL[vN].A(),width/2)
          xC2,yC2 = GM.getTargPoint(xT,yT,toboxRL[vN].opposite().A(),width/2)
          # get the bottom front & back points
          xC3,yC3 = GM.getTargPoint(xB,yB,toboxRL[vN].opposite().A(),width/2)
          xC4,yC4 = GM.getTargPoint(xB,yB,toboxRL[vN].A(),width/2)
          retD[vert] = GM.Polygon([xC1,xC2,xC3,xC4],[yC1,yC2,yC3,yC4])
    return retD

  # input: the angle of the polygon off the vertcal axis
  def rectangleImg(self,angle,polygon):
    """Returns a the sub-image defined by the polygon (assumes a
    rectangular shape).  Uses "angle" to righten the specified
    rectangle appropriately (e.g. is it a wide rectangle leaning
    to the left or a narrow rectangle leaning to the right?)
    """
    aortAng = math.radians(180) - angle
    sideNs = range(4)
    if polygon.numSides()!=4: raise ValueError("aortaRectImg REQ a rectangle")
    apXYL = map(lambda n: polygon.corner(n), sideNs)
    apXL,apYL = map(lambda i: i[0], apXYL),map(lambda i: i[1], apXYL)
    # get the center of the aorta rectangle to rotate around
    xC,yC = int(np.mean(apXL)),int(np.mean(apYL))
    # rotate the image and the polygon around the center
    M = cv2.getRotationMatrix2D((xC, yC), math.degrees(aortAng), 1.0)
    rows,cols = self._img.shape[:2]
    rotImg = cv2.warpAffine(self._img, M, (cols,rows))#self._img.shape[:2])
    vcXL,vcYL = map(lambda x: x-xC, apXL),map(lambda y: y-yC, apYL)
    rvcXL = map(lambda n: vcXL[n]*math.cos(-aortAng) - vcYL[n]*math.sin(-aortAng), sideNs)
    rvcYL = map(lambda n: vcYL[n]*math.cos(-aortAng) - vcXL[n]*math.sin(-aortAng), sideNs)
    rtXL,rtYL = map(lambda x: x+xC, rvcXL),map(lambda y: y+yC, rvcYL)
    rXmin,rXmax,rYmin,rYmax = min(rtXL),max(rtXL),min(rtYL),max(rtYL)
    # return the cropped rotated image
    return rotImg[int(rYmin):int(rYmax), int(rXmin):int(rXmax)]

  # HELPER FUNCTIONS

  def _makeAiBoxes(self):
    """Applies both of the object-detection models and stores
    the returned boxes as a list
    """
    if not(self._img is None) and not(self._hasBx):
      self._vertBL = self._modSet['vertDet'].getBoxes(self._img)
      self._vertExBL = []
      self._baseBL = self._modSet['baseDet'].getBoxes(self._img)
      self._baseExBL = []
    self._hasBx = True

  def _nForL4(self):
    """Uses the L4/L5 box to determine which vertebra in
    the sorted list of vertebra boxes is L4.  Looks for the box
    whose midpoint is the closest to the midpoint of the top half
    of the L4/L5 box of those boxes whose midpoints are within the
    L4/L5 box.
    RETURNS that list index (or -1 if no vertebra overlaps the box)
    """
    if not(self._isRefined): self.refineSpineTrace()
    xysL = []
    for b in GM.sortBoxesOnAxis(self._vertBL):
      x,y = b.midpoint()
      xysL.append( (x,y,b.score()) )
    if len(xysL)==0: return -1
    xysnL = map(lambda n: (xysL[n][0],xysL[n][1],xysL[n][2],n),
                range(len(xysL)))
    # get L4/L5 box as "bBox"
    if not(self._hasBx): self._makeAiBoxes()
    sbL = [(b.score(),b) for b in self._baseBL]
    bBox = max(sbL)[1]
    def ovlBase(xysn):
      x,y,s,n = xysn
      if x > bBox.xMax() or x < bBox.xMin(): return False
      if y > bBox.yMax() or y < bBox.yMin(): return False
      return True
    xysnLF = filter(ovlBase, xysnL)
    if len(xysnLF)==0: return -1
    elif len(xysnLF)==1: x4,y4,s4,n4 = xysnLF[0]
    else:
      yMid = (bBox.yMin()+bBox.yMax())/2
      # remember, the TOP of the picture is the lowest Y coord
      topHalfB = GM.Box(bBox.xMin(),bBox.yMin(),bBox.xMax(),yMid,0)
      xTM,yTM = topHalfB.midpoint()
      dxysL = [(GM.getDist(x,y,xTM,yTM),x,y,s,n) for x,y,s,n in xysnLF]
      d,x4,y4,s4,n4 = min(dxysL)
    return n4

  def _boxScoreFilter(self,goodL,badL,minScore):
    """Moves boxes from goodL to badL if their scores are below minScore"""
    moveL = filter(lambda b: b.score() < minScore, goodL)
    for b in moveL: goodL.remove(b)
    badL.extend(moveL)

  def _evaluateVerts(self):
    """Applies the is-this-vertebra-ok model to vertebra box list
    and moves any vertebra not deemed 'ok' to the trash.
    """
    h,w = self._img.shape[:2]
    moveL = []
    for b in self._vertBL:
      x,y = b.midpoint()
      if not(self._getVertOkVal(x,y,True)): moveL.append(b)
    for b in moveL: self._vertBL.remove(b)
    self._vertExBL.extend(moveL)    

  def _getVertOkVal(self,x,y,getBool):
    """helper for "_evaluateVerts": THIS function actually
    invokes the model.
    """
    bord,circ,colorCirc = 60,5,(0,250,0)
    h,w = self._img.shape[:2]
    # limited to the edges of the full image
    xMn,xMx = max([x-bord,0]),min([x+bord,w])
    yMn,yMx = max([y-bord,0]),min([y+bord,h])
    subImg = np.copy(self._img[yMn:yMx,xMn:xMx,:])
    # re-define coords for sub-image
    x,y = x-xMn,y-yMn
    cv2.circle(subImg,(x,y),circ,colorCirc,thickness=3)
    if getBool: return self._modSet['vertOk'].isOk(subImg)
    else: return self._modSet['vertOk'].scoreOk(subImg)    

  def _applySpineAngleFilter(self):
    """Eliminates vertebra that would create a sharp angle
    in the spine.  Moves them from the active vertebra list
    (_vertBL) to the trash vertebra list (_vertExBL).
    """
    # declare oldLength to be +1 to force a single cycle
    newLen,oldLen = len(self._vertBL),len(self._vertBL)+1
    # I need three in order to measure an angle
    while newLen < oldLen and newLen > 2:
      # update "newLength" at the end
      oldLen = newLen
      bL = GM.sortBoxesOnAxis(self._vertBL)
      vbpL = map(lambda n: VertebraBoxPair(bL[n-1],bL[n]), range(1,len(bL)))
      segPaL = map(lambda n: (GM.minAngDiff(vbpL[n-1].angle(),vbpL[n].angle()), n),
                   range(1,len(vbpL)) )
      maxAng,maxAngN = max(segPaL)
      if maxAng > self._maxAngV:
        doRemove = True # guarantees def. of "remBlock"
        if maxAngN == 1:
          if len(bL) < 4: doRemove = False
          else:
            # compare getting rid of second or first vertebra
            refAngle = vbpL[2].angle()
            optA = VertebraBoxPair(bL[0],bL[2]).angle()
            optB = vbpL[1].angle()
            if GM.minAngDiff(refAngle,optA) < GM.minAngDiff(refAngle,optB): remBlock = bL[1]
            else: remBlock = bL[0]
        elif maxAngN == len(vbpL) - 1:
          if maxAngN < 2: doRemove = False
          else:
            # compare getting rid of maxAngN or maxAngN + 1
            #               (second-to-final or final)
            refAngle = vbpL[maxAngN-2].angle()
            optA = VertebraBoxPair(bL[maxAngN-1],bL[maxAngN+1]).angle()
            optB = vbpL[maxAngN-1].angle()
            if GM.minAngDiff(refAngle,optA) < GM.minAngDiff(refAngle,optB): remBlock = bL[-2]
            else: remBlock = bL[-1]
        else: remBlock = bL[maxAngN]
        if doRemove:
          self._vertBL.remove(remBlock)
          self._vertExBL.append(remBlock)
      newLen = len(self._vertBL)

  def _adjustBox(self,box,xR,yR):
    """A convenience for changing the position of a box.  this
    is useful for the object-detector models that work on sub-images
    (adjusting the coordinates back to those of the full image).
    """
    xMin,xMax = box.xMin()+xR, box.xMax()+xR
    yMin,yMax = box.yMin()+yR, box.yMax()+yR
    return GM.Box(xMin,yMin,xMax,yMax,box.score())

  def _addSkippedBoxes(self):
    """Detects gaps in the spine and fills in missing vertebrae
    (or deletes doubly-annotated vertebrae).
    """
    changed = False
    bL = GM.sortBoxesOnAxis(self._vertBL)
    h,w = self._img.shape[:2]
    bord,circ = 60,40
    colorLine,colorCirc = (0,250,0),(0,0,250)
    # generate an image for each pair
    removeL,addBoxL = [],[]
    for n in range(1,len(bL)):
      xA,yA = bL[n-1].midpoint()
      xB,yB = bL[n].midpoint()
      xPL = [xA-bord,xA+bord,xB-bord,xB+bord]
      yPL = [yA-bord,yA+bord,yB-bord,yB+bord]
      # limited to the edges of the full image
      xMn,xMx = max([min(xPL),0]),min([max(xPL),w])
      yMn,yMx = max([min(yPL),0]),min([max(yPL),h])
      subImg = np.copy(self._img[yMn:yMx,xMn:xMx,:])
      # re-define coords for sub-image
      xA,yA,xB,yB = xA-xMn,yA-yMn,xB-xMn,yB-yMn
      cv2.line(subImg,(xA,yA),(xB,yB),colorLine,thickness=3)
      cv2.circle(subImg,(xA,yA),circ,colorCirc,thickness=2)
      cv2.circle(subImg,(xB,yB),circ,colorCirc,thickness=2)
      label = self._modSet['gapClass'].getClasses(subImg).best()
      if label=='adjacent': pass
      elif label=='double':
        scNm1 = self._getVertOkVal(xA+xMn,yA+yMn,False)
        scN = self._getVertOkVal(xB+xMn,yB+yMn,False)
        if scNm1 < scN: removeL.append(bL[n-1])
        else: removeL.append(bL[n])
        changed = True
      elif label=='skipone':
        fillBL = self._modSet['fill1Det'].getBoxes(subImg)
        fillBL = [(b.score(),b) for b in fillBL]
        newBS,newB = max(fillBL)
        if newBS >= self._minScP:
          addBoxL.append(self._adjustBox(newB,xMn,yMn))
          changed = True
      elif label=='skipmulti':
        fillBL = self._modSet['fillMuDet'].getBoxes(subImg)
        fillBL = filter(lambda b: b.score() >= self._minScP, fillBL)
        fillBL = [(b.score(),b) for b in fillBL]
        fillBL.sort()
        fillBL.reverse() # high-score first
        for fs,fb in fillBL[:2]:
          addBoxL.append(self._adjustBox(fb,xMn,yMn))
          changed = True
    for b in removeL:
      # need to do this because item could be removed 2X
      if b in self._vertBL:
        self._vertBL.remove(b)
        self._vertExBL.append(b)
    self._vertBL.extend(addBoxL)
    return changed

  def _addBottomBoxes(self):
    """Iteratively determines if a vertebra is missing from the
    bottom of the spine and extends the spine downwards if necessary
    """
    changed = False
    # copies the method from "createSpinePairImgs.py"
    # nb143 p165
    bL = GM.sortBoxesOnAxis(self._vertBL)
    if len(bL) > 0:
      bottom = bL[0]
      h,w = self._img.shape[:2]
      top,bord,circ = 60,100,40
      colorCirc,colorDot = (0,0,250),(0,250,0)
      changed = True # progress was made AND is needed
      for n in range(6):
        if changed:
          changed = False
          x,y = bottom.midpoint()
          xMn,xMx = max([x-bord,0]),min([x+bord,w])
          yMn,yMx = max([y-top,0]),h
          subImg = np.copy(self._img[yMn:yMx,xMn:xMx,:])
          # re-define coords for sub-image
          x,y = x-xMn,y-yMn
          cv2.circle(subImg,(x,y),circ,colorCirc,thickness=2)
          cv2.line(subImg,(x,y),(x,y),colorDot,thickness=5)
          if not(self._modSet['baseOk'].isOk(subImg)):
            fillBL = self._modSet['fillBotDet'].getBoxes(subImg)
            fillBL = [(b.score(),b) for b in fillBL]
            if len(fillBL) >0:
              newBS,newB = max(fillBL)
              newB = self._adjustBox(newB,xMn,yMn)
              if newBS >= self._minScP:
                # make sure the new box is at the bottom
                testBL = map(lambda i:i, self._vertBL)
                testBL.append(newB)
                testBL = GM.sortBoxesOnAxis(testBL)
                if testBL[0]==newB:
                  self._vertBL.append(newB)
                  bottom = newB
                  # only continue if L4 not reached
                  #changed = (self._nForL4() == -1)
                  # MODIFIED: I want L5 (so 0 => keep going)
                  changed = (self._nForL4() < 1)
    

class VertebraBoxPair:
  """Sometimes, the edge between two vertebrae is the data
  object of interest, not the nodes of the vertebrae themselves.
  This class is for those cases.
  """
  def __init__(self,lowerBox,upperBox):
    self._lower = lowerBox
    self._upper = upperBox
  def length(self):
    xA,yA = self._lower.midpoint()
    xB,yB = self._upper.midpoint()
    return getDist(xA,yA,xB,yB)
  def angle(self):
    xA,yA = self._lower.midpoint()
    xB,yB = self._upper.midpoint()
    horz = xB - xA
    vert = yB - yA
    # this order makes it degrees from vertical
    return math.atan2(horz,vert)
  def midpoint(self):
    xA,yA = self._lower.midpoint()
    xB,yB = self._upper.midpoint()
    return np.mean([xA,xB]),np.mean([yA,yB])
  def lowBox(self): return self._lower
  def highBox(self): return self._upper


# MODELS THAT I WANT TO USE PERMANENTLY;
# the path will be different if this is run from
# this script versus predict_aac_scores_2.py
if os.path.basename(os.getcwd())=='modules_mod2':
  modPath = '../models_mod2'
  permOldFileFile = 'original_files.txt'
else:
  modPath = 'models_mod2'
  permOldFileFile = 'modules_mod2/original_files.txt'
pMods = {}
for modName in ["vert","l4l5","okV","gap","fill1","fillM","okB","fillB"]:
  fullModName,fullLabelName = modName+'_mod', modName+'_label'
  pMods[fullModName] = os.path.join(modPath,fullModName+'.pb')
  pMods[fullLabelName] = os.path.join(modPath,fullModName+'.labels.txt')
permVerts = "L3-1,L4-0"


def main():
  # start the app
  ap = argparse.ArgumentParser()
  ap.add_argument("-i","--input_dir",
                  help="input directory of images to be cropped")
  ap.add_argument("-o","--output_dir",
                  help="output directory of cropped images")
  ap.add_argument("--original_resize",
                  help="resizes images as in Sethi et al: REQUIRES '<ID>.png' file name format.",
                  action='store_true')
  ap.add_argument("--no_resize",
                  help="prevents images from being re-sized: only use if you've checked your image sizes against the docs!",
                  action='store_true')
  args = vars(ap.parse_args())
  runAnalysis(args)



def runAnalysis(inputArgs):
  # so that I can modify the resize option values
  args = {}
  args.update(inputArgs)
  if not(args.has_key('original_resize')): args['original_resize'] = False
  if not(args.has_key('no_resize')): args['no_resize'] = False
  if args['original_resize'] and args['no_resize']:
    raise ValueError('original_resize and no_resize options cannot BOTH be selected')
  
  # set up all of the models
  modSet = {}
  modSet['vertDet'] = ML.TfObjectDetector(pMods["vert_mod"],pMods["vert_label"],1)  
  modSet['baseDet'] = ML.TfObjectDetector(pMods["l4l5_mod"],pMods["l4l5_label"],1)  
  modSet['vertOk'] = ML.TfBooleanApplyer(pMods["okV_mod"],pMods["okV_label"],'ok')
  modSet['gapClass'] = ML.TfClassApplyer(pMods["gap_mod"],pMods["gap_label"])
  modSet['fill1Det'] = ML.TfObjectDetector(pMods["fill1_mod"],pMods["fill1_label"],1)
  modSet['fillMuDet'] = ML.TfObjectDetector(pMods["fillM_mod"],pMods["fillM_label"],1)
  modSet['baseOk'] = ML.TfBooleanApplyer(pMods["okB_mod"],pMods["okB_label"],'ok')
  modSet['fillBotDet'] = ML.TfObjectDetector(pMods["fillB_mod"],pMods["fillB_label"],1)

  resizeOpt = 'default'
  if args['original_resize']: resizeOpt = 'original'
  if args['no_resize']: resizeOpt = 'none'
  
  vertL = map(lambda i: i.split('-'), permVerts.split(','))
  vertL = [(i[0],int(i[1])) for i in vertL]
  imgWriter = ImgGenerator(args["input_dir"],modSet,args["output_dir"],vertL,resizeOpt)
  nWrit = 0
  n = 0
  while imgWriter.hasImage():
    n += 1
    if n % 10 == 0:
      sys.stdout.write('.')
      sys.stdout.flush()
    if n % 200 == 0: sys.stdout.write('\n')
    if imgWriter.makeImage(): nWrit += 1
  print n,'read,',nWrit,'written.'
  print imgWriter.resizeSummary()

if __name__ == "__main__": main()
