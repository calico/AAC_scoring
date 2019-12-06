import tensorflow as tf
from utils import label_map_util  
import numpy as np
import geometry_tools as GM
import cv2

# separate out the box-drawing
# this class uses code from:
# https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/
class TfObjectDetector:
  def __init__(self,existingModelFile,categoryFile,maxNumClasses):
    self._modFile = existingModelFile
    self._catFile = categoryFile
    # ## Load a (frozen) Tensorflow model into memory.
    self._detection_graph = tf.Graph()
    with self._detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self._modFile, 'rb') as fid:
        serialized_graph = fid.read()
        print self._modFile
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(self._catFile)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=maxNumClasses,
                                                                use_display_name=True)
    self._category_index = label_map_util.create_category_index(categories)
    self._sess = tf.Session(graph=self._detection_graph)
  def getBoxes(self,image):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = self._sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
    h,w,ch = image.shape
    bL,scL,numB = boxes[0],scores[0],num_detections[0]
    boxL = []
    for n in range(numB):
       yA,yB = int(bL[n][0]*h),int(bL[n][2]*h)
       xA,xB = int(bL[n][1]*w),int(bL[n][3]*w)
       boxL.append(GM.Box(xA,yA,xB,yB,scL[n]))
    return boxL


class NullBooleanApplyer:
  def __init__(self): pass
  def isOk(self,image): return True
class TfBooleanApplyer:
  def __init__(self,existingModelFile,categoryFile,okLabel,minScr=0.5):
    self._classApp = TfClassApplyer(existingModelFile,categoryFile)
    self._okLabel = okLabel
    self._minScr = minScr
  def scoreOk(self,image):
    result = self._classApp.getClasses(image)
    return result.score(self._okLabel)
  def isOk(self,image):
    return self.scoreOk(image) >= self._minScr



class NullClassApplyer:
  def __init__(self,nullClass): self._class = nullClass
  def getClass(self,image): return TfClassResult( [(self._class, 1.0)] )
class TfClassApplyer:
  def __init__(self,existingModelFile,categoryFile):
    self._modFile = existingModelFile
    self._catFile = categoryFile
    proto_as_ascii_lines = tf.gfile.GFile(categoryFile).readlines()
    self._labels = map(lambda i: i.rstrip(), proto_as_ascii_lines)
    # ## Load a (frozen) Tensorflow model into memory.
    self._detection_graph = tf.Graph()
    with self._detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self._modFile, 'rb') as fid:
        serialized_graph = fid.read()
        print self._modFile
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    self._sess = tf.Session(graph=self._detection_graph)
  def getClasses(self,image,spCl=None):
    # get the image tensor so I can re-size the image appropriately
    image_tensor = self._detection_graph.get_tensor_by_name('Placeholder:0')
    image_resized = cv2.resize(image,dsize=tuple(image_tensor.shape[1:3]))
    image_np_expanded = np.expand_dims(image_resized, axis=0)
    image_np_expanded = image_np_expanded.astype(np.float32)
    image_np_expanded /= 255
    answer_tensor = self._detection_graph.get_tensor_by_name('final_result:0')
    # Actual detection.
    (answer_tensor) = self._sess.run([answer_tensor],
                                     feed_dict={image_tensor: image_np_expanded})
    results = np.squeeze(answer_tensor)
    results = [(results[n],self._labels[n]) for n in range(len(self._labels))]
    return TfClassResult(results)

class TfClassResult:
  # takes a list of score,label tuples
  def __init__(self,results):
    self._rD = {}
    for s,lb in results: self._rD[lb] = s
    self._lbmx = max(results)[1]
  def best(self): return self._lbmx
  def score(self,lb): return self._rD[lb]
  def labels(self): return self._rD.keys()


