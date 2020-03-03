import os


class ImageIterator:
  def __init__(self,hostDir):
    # check that the host dir exists
    if not(os.path.isdir(hostDir)):
      raise ValueError("host dir doesn't exist")
    self._hostD = os.path.abspath(hostDir)
    self._isSrt = False
  def getCatagories(self):
    return map(lambda i:i, self._catL)
  def initiateSort(self):
    self._isSrt = True
    cL = os.listdir(self._hostD)
    cL = list(map(lambda i: os.path.join(self._hostD,i), cL))
    cL = list(filter(lambda i: os.path.isfile(i), cL))
    cL = list(filter(lambda i: i.split('.')[-1].lower()!='txt', cL))
    cL = list(filter(lambda i: i.split('.')[-1].lower()!='tsv', cL))
    cL = list(filter(lambda i: i.split('.')[-1].lower()!='csv', cL))
    self._candL = cL
    self._candN = 0
    if self._candN==len(self._candL): self._isSrt = False
  def isSorting(self): return self._isSrt
  def getImgFile(self):
    if not(self._isSrt): raise ValueError("NOT SORTING")
    return self._candL[self._candN]
  def quit(self): pass
  def moveToNext(self):
    if not(self._isSrt): raise ValueError("NOT SORTING")
    self._candN += 1
    if self._candN==len(self._candL): self._isSrt = False    
  def goBackOne(self):
    if not(self._isSrt): raise ValueError("NOT SORTING")
    if self._candN > 0:
      self._candN -= 1
      if self._candN==len(self._candL): self._isSrt = False    

