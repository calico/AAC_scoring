'''
Extract Aortic region from X-ray image given Pelvis and Vertebrae segmentations

Author: Jagadish Venkataraman
Date: 4/30/2019
'''
from typing import Union, List
import numpy as np
from scipy import ndimage

class ExtractAorticRegion(object):
    def __call__(self,
                 input_image: np.ndarray,
                 vertebrae_mask: np.ndarray,
                 pelvis_mask: np.ndarray,
                 vertebrae_span: int=4,
                 aorta_width: int=150,
                 aorta_offset: int = 20,
                 spinal_curve_deg: int = 3,
                 spinal_correction_threshold: int = 10,
                 min_vertebrae_count: int = 3,
                 max_area_ratio_threshold: float = 0.7,
                 centroid_deviation_percentage: float = 0.25,
                 blur_radius: float=1.,
                 intensity_threshold: float=200.) -> None:

        self.input_image = input_image
        self.vertebrae_mask = vertebrae_mask
        self.pelvis_mask = pelvis_mask
        self.vertebrae_span = vertebrae_span
        self.blur_radius = blur_radius # pixels
        self.intensity_threshold = intensity_threshold # intensity
        self.aorta_width = aorta_width
        self.aorta_offset = aorta_offset
        self.spinal_curve_deg = spinal_curve_deg
        self.spinal_correction_threshold = spinal_correction_threshold
        self.min_vertebrae_count = min_vertebrae_count
        self.centroid_deviation_percentage = centroid_deviation_percentage
        self.max_area_ratio_threshold = max_area_ratio_threshold


        self.start_x_offset = None
        self.end_x_offset = None
        self.num_vertebrae = 0
        self.vertebrae_centroids = None
        self.vertebrae_boxes = []
        self.num_pelvis = 0
        self.pelvis_centroids = None
        self.labeled_vertebrae = None
        self.labeled_pelvis = None
        self.combined_centroids = None
        self.labeled_combine = None
        self.aorta_left = None
        self.aorta_right = None
        self.aorta_top_left_x = None
        self.aorta_top_left_y = None
        self.aorta_bottom_right_x = None
        self.aorta_bottom_right_y = None
        self.aortic_region = None
        self.aortic_region_mask = None
        self.vertebrae_sizes = []
        self.spinal_curve_coeffs = None
        self.image_shape = self.input_image.shape
        self.median_H = None
        self.median_W = None

        # sequence of methods
        self.get_vertebrae_centroids()
        if self.num_vertebrae > 2:
            self.get_pelvis_centroids()
            self.get_combined_centroids()
            self.get_vertebrae_sizes()
            self.fill_in_missing_vertebrae()
            self.get_aortic_offsets()
            self.get_aortic_region()


    def find_centroids(self, mask):
        '''
        Method to find centroids of the various regions
        '''
        m,n = mask.shape
        r,c = np.mgrid[:m,:n]
        count = np.bincount(mask.ravel())
        nonzero_idx = np.where(count != 0)
        centroid_row = np.bincount(mask.ravel(),r.ravel())[nonzero_idx]/count[nonzero_idx]
        centroid_col = np.bincount(mask.ravel(),c.ravel())[nonzero_idx]/count[nonzero_idx]
        centroids = np.round(np.c_[centroid_row, centroid_col]).astype(np.int32)
        # remove background centroid
        centroids = np.delete(centroids, 0, 0)

        return centroids


    def filter_and_label(self, mask: np.ndarray, biggest_region: bool=False, min_y: int=None) -> List[Union[np.ndarray, int]]:
        '''
        Method to lpf the masks, label them and identify centroids
        '''
        # filtered mask
        filtered_mask = ndimage.gaussian_filter(mask, self.blur_radius)

        # count the regions
        labeled_mask, count = ndimage.label(filtered_mask > self.intensity_threshold)

        # find and eliminate regions above min value
        if min_y is not None:
            labeled_mask[:min_y,:] = 0
            # identify the number of different regions below the lowest vertebrae
            uniq_labels = list(np.unique(labeled_mask))
            uniq_labels = [x for x in uniq_labels if x != 0]
            count = len(uniq_labels)

        if count > 0:

            # areas of the regions
            areas = np.bincount(labeled_mask.ravel())[1:]

            if biggest_region:
                # return the biggest region only
                max_idx = np.argmax(areas)
                # all other regions
                idx = [i for i, _ in enumerate(areas) if i != max_idx and _ != 0]
            else:
                # ignore regions that are smaller than desired threshold
                max_area = areas.max()
                # determine regions that fall below min size requirement and merge them with the background
                idx = [i for i, _ in enumerate(areas) if _ < max_area*self.max_area_ratio_threshold]

            for i in idx:
                labeled_mask[labeled_mask == (i+1)] = 0
                count -= 1

        return [labeled_mask, count]


    def get_vertebrae_centroids(self) -> None:
        '''
        Method to determine vertebrae centroids
        '''
        self.labeled_vertebrae, self.num_vertebrae = self.filter_and_label(self.vertebrae_mask)

        if self.num_vertebrae > 0:
            self.vertebrae_centroids = self.find_centroids(self.labeled_vertebrae)
            self.vertebrae_centroids = self.vertebrae_centroids[self.vertebrae_centroids[:,0].argsort()]
            if self.vertebrae_span < self.vertebrae_centroids.shape[0]:
                self.vertebrae_centroids = self.vertebrae_centroids[-self.vertebrae_span:,:]
            else:
                self.vertebrae_span = self.vertebrae_centroids.shape[0]


    def get_pelvis_centroids(self) -> None:
        '''
        Method to determine pelvis centroid(s)
        '''
        if self.pelvis_mask is not None and self.vertebrae_centroids is not None:
            self.labeled_pelvis, self.num_pelvis = self.filter_and_label(self.pelvis_mask, biggest_region=True, min_y=self.vertebrae_centroids[-1,0])

        if self.num_pelvis > 0:
            self.pelvis_centroids = self.find_centroids(self.labeled_pelvis)
            self.pelvis_centroids = self.pelvis_centroids[self.pelvis_centroids[:,0].argsort()]


    def get_spinal_curvature(self) -> None:
        '''
        Method to fit a polynomial to the spinal curvature
        '''
        x_arr = np.asarray([c[1] for c in self.combined_centroids])
        y_arr = np.asarray([c[0] for c in self.combined_centroids])

        # p(y) for x
        self.spinal_curve_coeffs = np.polyfit(y_arr, x_arr, self.spinal_curve_deg)
        self.p_x = np.poly1d(self.spinal_curve_coeffs)
        x_arr_est = np.round(self.p_x(y_arr))
        # correct along x
        for idx, (x, x_est) in enumerate(zip(list(x_arr), list(x_arr_est))):
            if abs(x - x_est) > self.spinal_correction_threshold:
                self.combined_centroids[idx][1] = x_est


    def get_combined_centroids(self) -> None:
        '''
        Method to combine the vertebrae and pelvis centroids
        '''
        if self.labeled_vertebrae is not None:
            if self.num_pelvis > 0:
                self.labeled_combine = self.labeled_pelvis.copy()
                # change the label for the pelvis
                self.labeled_combine[self.labeled_pelvis != 0] = self.num_vertebrae + 1
                # combine
                self.labeled_combine = np.add(self.labeled_combine, self.labeled_vertebrae)
                self.combined_centroids = np.vstack([self.vertebrae_centroids, self.pelvis_centroids])
                self.combined_centroids = self.combined_centroids[self.combined_centroids[:,0].argsort()]
            else:
                self.labeled_combine = self.labeled_vertebrae.copy()
                self.combined_centroids = self.vertebrae_centroids.copy()
                self.combined_centroids = np.append(self.combined_centroids, [[self.combined_centroids[-1, 0], self.combined_centroids[-1, 1]]], axis=0)


    def get_aortic_region(self) -> None:
        '''
        Method to get Aortic region
        '''

        centroids_y = self.combined_centroids[(-1-self.vertebrae_span):-1,0]
        centroids_y = np.expand_dims(centroids_y, axis=1)
        centroids_x = 0*self.combined_centroids[(-1-self.vertebrae_span):-1,1]
        centroids_x = np.expand_dims(centroids_x, axis=1)
        self.aorta_left = np.hstack([centroids_y, centroids_x]) + np.hstack([np.zeros(self.start_x_offset[(-self.vertebrae_span):,:].shape).astype(int), self.start_x_offset[(-self.vertebrae_span):,:]])
        self.aorta_right = np.hstack([centroids_y, centroids_x]) + np.hstack([np.zeros(self.start_x_offset[(-self.vertebrae_span):,:].shape).astype(int), self.end_x_offset[(-self.vertebrae_span):,:]])

        self.aorta_top_left_x = (min(np.min(self.aorta_left[:,1]), np.min(self.aorta_right[:,1]))).astype(int)
        self.aorta_top_left_y = (min(np.min(self.aorta_left[:,0]), np.min(self.aorta_right[:,0])) - 0*self.vertebrae_sizes[0][0]//2).astype(int)

        self.aorta_bottom_right_x = (max(np.max(self.aorta_left[:,1]), np.max(self.aorta_right[:,1]))).astype(int)
        self.aorta_bottom_right_y = (max(np.max(self.aorta_left[:,0]), np.max(self.aorta_right[:,0])) + 0*self.vertebrae_sizes[-1][0]//2).astype(int)

        # create mask of aortic region
        self.aortic_region_mask = np.zeros(self.image_shape)

        for idx in range(self.vertebrae_span-1):
            if idx == 0:
                y_desired = np.arange(self.aorta_top_left_y, self.aorta_left[idx+1, 0]).astype(int)
                x_left_desired = np.round(np.interp(y_desired,
                                                    self.aorta_left[idx:idx+2,0],
                                                    self.aorta_left[idx:idx+2,1])).astype(int)
                x_right_desired = np.round(np.interp(y_desired,
                                                     self.aorta_right[idx:idx+2,0],
                                                     self.aorta_right[idx:idx+2,1])).astype(int)
            elif idx == self.vertebrae_span-2:
                y_desired = np.arange(self.aorta_left[idx, 0], self.aorta_bottom_right_y).astype(int)
                x_left_desired = np.round(np.interp(y_desired,
                                                    self.aorta_left[idx:,0],
                                                    self.aorta_left[idx:,1])).astype(int)
                x_right_desired = np.round(np.interp(y_desired,
                                                     self.aorta_right[idx:,0],
                                                     self.aorta_right[idx:,1])).astype(int)
            else:
                y_desired = np.arange(self.aorta_left[idx, 0], self.aorta_left[idx+1, 0]).astype(int)
                x_left_desired = np.round(np.interp(y_desired,
                                                    self.aorta_left[idx:idx+2,0],
                                                    self.aorta_left[idx:idx+2,1])).astype(int)
                x_right_desired = np.round(np.interp(y_desired,
                                                     self.aorta_right[idx:idx+2,0],
                                                     self.aorta_right[idx:idx+2,1])).astype(int)

            y_desired = np.clip(y_desired, 0, self.image_shape[0]-1).astype(np.int32)
            x_left_desired = np.clip(x_left_desired, 0, self.image_shape[1]-1).astype(np.int32)
            x_right_desired = np.clip(x_right_desired, 0, self.image_shape[1]-1).astype(np.int32)

            for y, x_left, x_right in zip(y_desired, x_left_desired, x_right_desired):
                self.aortic_region_mask[min(y, self.aortic_region_mask.shape[0]), x_left:x_right] = 1


        # ensure that the aortic region mask does not encroach on any non-zero labels in the labeled_combine image
        self.aortic_region_mask = np.multiply(self.aortic_region_mask, np.logical_not(self.labeled_combine))

        # Apply mask to image
        self.aortic_region = np.multiply(self.input_image[self.aorta_top_left_y:self.aorta_bottom_right_y,
                                              self.aorta_top_left_x:self.aorta_bottom_right_x],
                                         self.aortic_region_mask[self.aorta_top_left_y:self.aorta_bottom_right_y,
                                              self.aorta_top_left_x:self.aorta_bottom_right_x])
        # replace black pixels with min values in the aortic region
        self.aortic_region[np.where(self.aortic_region == 0.)] = np.median(self.input_image[self.aorta_top_left_y:self.aorta_bottom_right_y,
                                              self.aorta_top_left_x:self.aorta_bottom_right_x])


    def get_vertebrae_sizes(self) -> None:
        '''
        Method to determine vertebrae sizes and also create "convex hulls" around the vertebrae
        '''
        x_arr = [c[1] for c in self.vertebrae_centroids]
        y_arr = [c[0] for c in self.vertebrae_centroids]

        for (x, y) in zip(x_arr, y_arr):
            label = self.labeled_vertebrae[y, x]
            locs = np.where(self.labeled_vertebrae == label)
            H = np.max(locs[0]) - np.min(locs[0])
            W = np.max(locs[1]) - np.min(locs[1])
            self.vertebrae_sizes.append([H, W])
            # expand the labeled combine to contain boxes covering the convex hull of the vertebrae
            start_y = np.min(locs[0])
            end_y = np.max(locs[0])
            start_x = np.min(locs[1])
            end_x = np.max(locs[1]) + self.aorta_offset
            self.labeled_combine[start_y:end_y, start_x:end_x] = label
            # store the vertebrae boxes
            self.vertebrae_boxes.append([start_y, end_y, start_x, end_x])


        # to compensate for broken segmentations, snap H,W smaller than threshold below median to median value
        self.median_H = np.round(np.median(np.asarray([x[0] for x in self.vertebrae_sizes])))
        self.median_W = np.round(np.median(np.asarray([x[1] for x in self.vertebrae_sizes])))

        for idx, [H, W] in enumerate(self.vertebrae_sizes):
            if abs(H - self.median_H) > self.median_H*0.5:
                self.vertebrae_sizes[idx][0] = self.median_H
            if abs(W - self.median_W) > self.median_W*0.5:
                self.vertebrae_sizes[idx][1] = self.median_W

        # get spinal curvature and adjust the centroids deviating from the curve
        self.get_spinal_curvature()


    def get_aortic_offsets(self) -> None:
        '''
        Method to determine the offset of the Aorta from the vertebrae centroids
        '''

        # sort the box regions
        self.vertebrae_boxes = sorted(self.vertebrae_boxes, key=lambda z:z[0])

        vertebrae_width = np.asarray([x[1] for x in self.vertebrae_sizes]).T
        vertebrae_width = vertebrae_width[...,np.newaxis]
        self.start_x_offset = (0*vertebrae_width//2 + np.min([v[3] for v in  self.vertebrae_boxes[-self.vertebrae_span:]])).astype(int) # HACK
        self.end_x_offset = (0*vertebrae_width//2 + np.max([v[3] for v in  self.vertebrae_boxes[-self.vertebrae_span:]]) + self.aorta_width).astype(int)

        # go through the vertebrae boxes and mask out the combined_label regions between vertebrae
        for idx in range(len(self.vertebrae_boxes)-1):
            start_x = min(self.vertebrae_boxes[idx][2], self.vertebrae_boxes[idx+1][2])
            end_x = min(self.vertebrae_boxes[idx][3], self.vertebrae_boxes[idx+1][3])
            start_y = self.vertebrae_boxes[idx][1]
            end_y = self.vertebrae_boxes[idx+1][0]
            self.labeled_combine[start_y:end_y, start_x:end_x] = -1


    def fill_in_missing_vertebrae(self) -> None:
        '''
        Method to fill in missing vertebrae segments based on distance estimates between successive vertebrae centroids - only if more than 2 vertebrae are segmented out
        '''
        if self.num_vertebrae >= self.min_vertebrae_count:
            # find distance between centroids
            distance_between_centroids = np.sqrt(np.linalg.norm(np.diff(self.combined_centroids, axis=0), axis=1))
            median_distance = np.median(distance_between_centroids)
            deviation_from_median = distance_between_centroids - median_distance
            start_idx = 0

            # loop over the values and identify deviations
            for value in deviation_from_median:
                if value > median_distance*self.centroid_deviation_percentage:
                    # number of missing vertebrae in the deviation
                    num_missing_vertebrae = np.ceil(value/median_distance).astype(np.int32)

                    # fill in centroids along the spinal curve
                    y_values_to_insert = np.round(np.linspace(self.combined_centroids[start_idx, 0], self.combined_centroids[start_idx+1, 0], num_missing_vertebrae+2))[1:-1]
                    x_values_to_insert = np.round(self.p_x(y_values_to_insert))

                    for idx in range(num_missing_vertebrae):
                        self.combined_centroids = np.insert(self.combined_centroids, start_idx+1, np.array([y_values_to_insert[idx], x_values_to_insert[idx]]), 0)
                        self.vertebrae_sizes = np.insert(self.vertebrae_sizes, start_idx+1, np.array([self.median_H, self.median_W]), 0)
                        # fill in boxes estimated for the missing vertebrae with label values
                        start_y = (y_values_to_insert[idx] - self.median_H//2).astype(int)
                        end_y = (y_values_to_insert[idx] + self.median_H//2).astype(int)
                        start_x = (x_values_to_insert[idx] - self.median_W//2).astype(int)
                        end_x = (x_values_to_insert[idx] + self.median_W//2 + self.aorta_offset).astype(int)
                        self.labeled_combine[start_y:end_y, start_x:end_x] = np.max(self.labeled_combine) + 1
                        self.vertebrae_boxes.append([start_y, end_y, start_x, end_x])

                    start_idx += num_missing_vertebrae

                start_idx += 1
